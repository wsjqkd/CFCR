import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BartForConditionalGeneration, ViTModel
from transformers.modeling_outputs import BaseModelOutput
import torch.nn.functional as F
from transformers import BartTokenizer
import numpy as np
import pickle
import os
from collections import OrderedDict
from scorer import rouge_score

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):

    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def post_process(hypothesis):
    hypothesis = hypothesis.replace('<s>', '')
    hypothesis = hypothesis.replace('<pad>', '')
    hypothesis = hypothesis.replace('</s>', '')
    hypothesis = hypothesis.replace("'s", " 's")
    hypothesis = hypothesis.strip()
    return hypothesis

class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False

class RegionSelector(nn.Module):
    def __init__(self, image_hidden_dim,  num_patches=49, hidden_dim=512):
        super().__init__()

        self.text_projection = nn.Linear(1, num_patches)  
        self.mlp = nn.Sequential(
            nn.Linear(768*2, image_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(image_hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )

    def forward(self, patch_features, text_feature):
        text_feature=text_feature.transpose(1,2)
        text_projected = self.text_projection(text_feature)
        text_projected = text_projected.transpose(1, 2)
        combined_features = torch.cat((patch_features, text_projected), dim=-1)
        importance_scores = self.mlp(combined_features)
        return importance_scores

class EntityObjectFusion(nn.Module):
    def __init__(self, hidden_size=768, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.W_e1 = nn.Linear(hidden_size, hidden_size)
        self.W_e2 = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, entity_embeds, useful_objects):

        attn_output, _ = self.mha(
            query=entity_embeds,  # H_Te
            key=useful_objects,   # D_Vo
            value=useful_objects  # D_Vo
        )  

        gate_input = self.W_e1(attn_output) + self.W_e2(entity_embeds)
        alpha_e = self.sigmoid(gate_input)  

        # Fused features: M_e = α_e · R_e + (1 − α_e) · H_Te
        fused_features = alpha_e * attn_output + (1 - alpha_e) * entity_embeds

        return fused_features


class CFCRSum(nn.Module):
    def __init__(self, config,train_texts):
        super().__init__()
        self.config = config
        self.device = config.device
        self.bart = BartForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=config.language_foundation_model)
        self.vit = ViTModel.from_pretrained(
            pretrained_model_name_or_path=config.vision_foundation_model)
        self.vitemb=self.vit.embeddings
        self.bart_tokenizer = BartTokenizer.from_pretrained(config.language_foundation_model)
        self.region_selector = RegionSelector(768, self.bart.config.hidden_size)
        self.fixed_hard_labels_dict = {}
        self.e1= self.config.e1
        self.e2= self.config.e2
        self.temperature = nn.Parameter(torch.tensor(0.07))            
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.entity_object_fusion = EntityObjectFusion(hidden_size=768, num_heads=8)
        self.train_texts = train_texts
        hidden_size=768
        self.projhead = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(hidden_size, hidden_size*2)),
                ('bn1', nn.BatchNorm1d(hidden_size*2)),
                ('relu1', nn.ReLU()),
                ('fc2', nn.Linear(hidden_size*2, hidden_size)),
                ('bn2', BatchNorm1dNoBias(hidden_size)),
            ]))
        self.projhead2 = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(hidden_size, hidden_size*2)),
                ('bn1', nn.BatchNorm1d(hidden_size*2)),
                ('relu1', nn.ReLU()),
                ('fc2', nn.Linear(hidden_size*2, hidden_size)),
                ('bn2', BatchNorm1dNoBias(hidden_size)),
            ]))       
    
    def save_hard_labels(self, hard_labels_all, num_regions_per_image, indices):
        assert hard_labels_all.shape[0] == len(indices), f"hard_labels_all batch size ({hard_labels_all.shape[0]}) does not match indices length ({len(indices)})"
        assert hard_labels_all.shape[1] == num_regions_per_image[0], f"hard_labels_all num_patches ({hard_labels_all.shape[1]}) does not match expected ({num_regions_per_image[0]})"

        for b, idx in enumerate(indices):
            hl = hard_labels_all[b].squeeze(-1)  # [50, 1] -> [50]
            self.fixed_hard_labels_dict[int(idx)] = hl.tolist()

        save_data = {
            'hard_labels': self.fixed_hard_labels_dict
        }
        with open('fixed_hard_labels0.01.pkl', 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Saved hard_labels for {len(self.fixed_hard_labels_dict)} ")

    def load_hard_labels(self):
        if os.path.exists('fixed_hard_labels0.01.pkl'):
            with open('fixed_hard_labels0.01.pkl', 'rb') as f:
                save_data = pickle.load(f)
            self.fixed_hard_labels_dict = save_data['hard_labels']
            print(f"Loaded hard_labels for {len(self.fixed_hard_labels_dict)} images")
        else:
            print("No hard labels file found")
        return self.fixed_hard_labels_dict

    def compute_contrastive_loss(self, summarization_features, global_image):

        batch_size = summarization_features.size(0)
        proj_summarization = self.projhead(summarization_features) 
        proj_image = self.projhead(global_image)  

        proj_summarization = F.normalize(proj_summarization, dim=-1)
        proj_image = F.normalize(proj_image, dim=-1)

        sim_matrix = torch.mm(proj_summarization, proj_image.t().contiguous())  
        sim_matrix = torch.exp(sim_matrix / self.temperature)

        pos_mask = torch.eye(batch_size, device=sim_matrix.device, dtype=torch.bool)  
        pos_sim = (sim_matrix * pos_mask).sum(dim=1)  

        neg_sim = sim_matrix.sum(dim=1) - pos_sim  

        # InfoNCE 
        loss = -torch.log(pos_sim / (neg_sim + pos_sim + 1e-8))  
        loss = loss.mean()

        return loss

    def compute_finecontrastive_loss(self, image_features, sentence_features):
        assert image_features.dim() == 3 and sentence_features.dim() == 3

        B, L1, H = image_features.shape
        _, L2, _ = sentence_features.shape

        image_flat = image_features.view(-1, H)  # [B*L1, H]
        sentence_flat = sentence_features.view(-1, H)  # [B*L2, H]

        proj_image = self.projhead2(image_flat)  # [B*L1, hidden_size]
        proj_sentence = self.projhead2(sentence_flat)  # [B*L2, hidden_size]
    
        proj_image = F.normalize(proj_image, dim=-1)  # [B*L1, hidden_size]
        proj_sentence = F.normalize(proj_sentence, dim=-1)  # [B*L2, hidden_size]

        logit_scale = self.logit_scale.exp().clamp(min=0.01, max=1.0)
        logits = logit_scale * (proj_sentence @ proj_image.T)  

        image_indices = torch.arange(B * L1, device=image_features.device) // L1  # [B*L1]
        text_indices = torch.arange(B * L2, device=sentence_features.device) // L2  # [B*L2]
        mask = (text_indices.unsqueeze(1) == image_indices.unsqueeze(0))  # [B*L2, B*L1]

        exp_logits = torch.exp(logits)
        pos_sum_t2i = (exp_logits * mask).sum(dim=1, keepdim=True)  
        all_sum_t2i = exp_logits.sum(dim=1, keepdim=True) 
        loss_t2i = -torch.log(pos_sum_t2i / (all_sum_t2i + 1e-8)).mean()

        return loss_t2i

    def forward(self, epoch,pixel_values, sentence_input_ids, sentence_attention_mask,
                summarization_input_ids, summarization_attention_mask, labels,
                sentence_entity_ids=None, sentence_entity_mask=None,
                summary_entity_ids=None, summary_entity_mask=None,
                useful_objects=None, indices=None,a=None, b=None):

        decoder_input_ids = shift_tokens_right(
            input_ids=labels,
            pad_token_id=self.bart.config.pad_token_id,
            decoder_start_token_id=self.bart.config.decoder_start_token_id)

        language_encoder_outputs = self.bart.model.encoder(
            input_ids=sentence_input_ids,
            attention_mask=sentence_attention_mask,
            return_dict=True)
        sentence_features = language_encoder_outputs.last_hidden_state

        vision_encoder_outputs = self.vitemb(
            pixel_values=pixel_values)
        patch_features = vision_encoder_outputs
        batch_size, num_patches, _ = patch_features.shape

        summarization_embeds = self.bart.model.encoder(
            input_ids=summarization_input_ids,
            attention_mask=summarization_attention_mask,
            return_dict=True )['last_hidden_state']
        summarization_features = summarization_embeds[:,0,:].detach()
        selector_loss = None
        hard_labels_all = None
        num_regions_per_image = [num_patches-1] * batch_size

        if self.e1< epoch <=self.e2:
            with torch.no_grad():  
                reference_texts = []
            for idx in indices:
                idx=idx-1
                reference_texts.append(self.train_texts[int(idx)]['summarization'])

            # complete image summary
            combined_input_full = torch.cat((patch_features[:, 1:, :], language_encoder_outputs.last_hidden_state), dim=1)
            encoder_outputs_full = BaseModelOutput(last_hidden_state=combined_input_full)
            full_summary_ids = self.bart.generate(encoder_outputs=encoder_outputs_full,
                                                 max_length=self.config.max_summarization_length,
                                                 num_beams=3)
            full_summaries = [post_process(hyp) for hyp in self.bart_tokenizer.batch_decode(full_summary_ids, skip_special_tokens=False)]
            full_rouge_scores = rouge_score(hypotheses=full_summaries, references=reference_texts,per_sample=True)
            full_rouge_avg = torch.tensor([(s['rouge-1']['f'] + s['rouge-2']['f'] + s['rouge-l']['f']) / 3 
                                          for s in full_rouge_scores], device=self.device)  

            effects = torch.zeros(batch_size, num_patches - 1, 1, device=self.device)  # [bs, 49, 1]
            group_ranges = [(0, 14), (14, 28), (28, 49)]  
            for group_idx, (start_idx, end_idx) in enumerate(group_ranges):
                mask = torch.ones(batch_size, num_patches - 1, 1, device=self.device)
                mask[:, start_idx:end_idx, :] = 0
                patch_feats_masked = patch_features[:, 1:, :] * mask  
                combined_input_without_i = torch.cat((patch_feats_masked, language_encoder_outputs.last_hidden_state), dim=1)
                encoder_outputs_without_i = BaseModelOutput(last_hidden_state=combined_input_without_i)
                stripped_summary_ids = self.bart.generate(encoder_outputs=encoder_outputs_without_i,
                                                         max_length=self.config.max_summarization_length,
                                                         num_beams=3)
                stripped_summaries = [post_process(hyp) for hyp in self.bart_tokenizer.batch_decode(stripped_summary_ids, skip_special_tokens=False)]
                stripped_rouge_scores = rouge_score(hypotheses=stripped_summaries, references=reference_texts, per_sample=True)
                stripped_rouge_avg = torch.tensor([(s['rouge-1']['f'] + s['rouge-2']['f'] + s['rouge-l']['f']) / 3 
                                                  for s in stripped_rouge_scores], device=self.device)
                group_CR = full_rouge_avg - stripped_rouge_avg
                effects[:, start_idx:end_idx, 0] = group_CR.unsqueeze(1).expand(-1, end_idx - start_idx)
            hard_labels_all = (effects > 0).float()
            self.save_hard_labels(hard_labels_all, num_regions_per_image, indices)

        # train selector
        if epoch > self.e1:
            if not self.fixed_hard_labels_dict:
                self.load_hard_labels()
            importance_scores_all = self.region_selector(patch_features[:, 1:, :],sentence_features[:,0,:].unsqueeze(1))
            hard_labels_all = torch.stack([torch.tensor(self.fixed_hard_labels_dict[int(idx)], dtype=torch.float32).to(self.device).unsqueeze(-1) 
                                          for idx in indices]) 
            
            bce=nn.BCELoss()
            selector_loss = bce(importance_scores_all, hard_labels_all)

        if epoch > self.e2 and self.fixed_hard_labels_dict:
            threshold = 0.5  
            useful_mask = (importance_scores_all > threshold).float()  
        else:
            useful_mask = torch.ones(batch_size, num_patches-1, 1, device=self.config.device) 

        masked_patch_features = patch_features[:, 1:, :] * useful_mask  # [bs, 50, 768]
        img_encoded = torch.mean(masked_patch_features, dim=1)  # [bs, 768]
 
        batch_size = sentence_entity_ids.size(0)
        entity_ids = sentence_entity_ids .view(batch_size, -1) 
        entity_mask = sentence_entity_mask.view(batch_size, -1)  
        entity_outputs = self.bart.model.encoder(
            input_ids=entity_ids,
            attention_mask=entity_mask,
            return_dict=True
        )
        entity_embeds = entity_outputs.last_hidden_state

        sentence_object_contrastive_loss = self.compute_finecontrastive_loss(useful_objects, summarization_embeds)
        image_sentence_contrastive_loss = self.compute_contrastive_loss(summarization_features, img_encoded)

        fine_features=self.entity_object_fusion(entity_embeds, useful_objects)

        last_hidden_state = torch.cat(
            tensors=(sentence_features, masked_patch_features,fine_features),dim=1)

        encoder_outputs = BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=None,
            attentions=None
        )
        outputs = self.bart.model(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            use_cache=False,
            return_dict=True
        )

        lm_logits = self.bart.lm_head(outputs['last_hidden_state'])
        lm_logits = lm_logits + self.bart.final_logits_bias.to(lm_logits.device)

        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(lm_logits.view(-1, self.bart.config.vocab_size), labels.view(-1))

        total_loss = (masked_lm_loss+
                     a*image_sentence_contrastive_loss+
                     b*sentence_object_contrastive_loss)
        if selector_loss is not None:
                    total_loss = (masked_lm_loss+selector_loss+
                                a*image_sentence_contrastive_loss+
                                b*sentence_object_contrastive_loss    )
        else:
            total_loss = (masked_lm_loss+
                     a*image_sentence_contrastive_loss+
                     b*sentence_object_contrastive_loss)           
                    
        return total_loss,masked_lm_loss,image_sentence_contrastive_loss,sentence_object_contrastive_loss,selector_loss
    
    def generate(self, pixel_values, sentence_input_ids, sentence_attention_mask,                
                 sentence_entity_ids=None, sentence_entity_mask=None,
                useful_objects=None, unused_objects=None,epoch=None):
        language_encoder_outputs = self.bart.model.encoder(
            input_ids=sentence_input_ids,
            attention_mask=sentence_attention_mask,
            return_dict=True)
        sentence_features = language_encoder_outputs.last_hidden_state

        vision_encoder_outputs = self.vitemb(
            pixel_values=pixel_values)
        
        patch_features = vision_encoder_outputs
        batch_size, num_patches, _ = patch_features.shape
        if epoch>self.e2:
            importance_scores_all = self.region_selector(patch_features[:, 1:, :], sentence_features[:,0,:].unsqueeze(1)) 
            useful_mask = (importance_scores_all > 0.5).float() 
            masked_patch_features = patch_features[:, 1:, :] * useful_mask
        else:
            masked_patch_features=patch_features[:, 1:, :]

        batch_size = sentence_entity_ids.size(0)
        entity_ids = sentence_entity_ids .view(batch_size, -1) 
        entity_mask = sentence_entity_mask.view(batch_size, -1)  

        entity_outputs = self.bart.model.encoder(
            input_ids=entity_ids,
            attention_mask=entity_mask,
            return_dict=True)
        entity_embeds = entity_outputs.last_hidden_state

        fine_features=self.entity_object_fusion(entity_embeds, useful_objects)

        last_hidden_state = torch.cat(
            tensors=(sentence_features, masked_patch_features,fine_features),dim=1)

        encoder_outputs = BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=None,
            attentions=None)

        return self.bart.generate(encoder_outputs=encoder_outputs, max_length=self.config.max_summarization_length, num_beams=self.config.num_beams)
