import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
import copy
from .Transformer import BidirectionalTransformer
import torch.nn.functional as F

#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        #print(x.shape)
        quant_z, indices, loss = self.vqgan.encode(x)
        #(_, _, indices) = loss
        # quant_z, _, (_, _, indices) = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices
        #raise Exception('TODO2 step1-1!')
        #return None
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda r: 1 - r
            #raise Exception('TODO2 step1-2!')
            #return None
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
            # raise Exception('TODO2 step1-2!')
            # return None
        elif mode == "square":
            return lambda r: 1 - r ** 2
            #raise Exception('TODO2 step1-2!')
            #return None
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        
        _, z_indices = self.encode_to_z(x) #ground truth
        # sos_tokens = torch.ones(x.shape[0], 1, dtype=torch.long, device=z_indices.device) * 1024

        r = math.floor(self.gamma(np.random.uniform()) * z_indices.shape[1])
        # print(r)
        # print(z_indices.shape[1])
        sample = torch.rand(z_indices.shape, device=z_indices.device).topk(r, dim=1).indices
        mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=z_indices.device)
        mask.scatter_(dim=1, index=sample, value=True)

        masked_indices = self.mask_token_id * torch.ones_like(z_indices, device=z_indices.device)

        a_indices = (~mask) * z_indices + (mask) * masked_indices
        # count = 0
        # for i in a_indices:
        #     sos_tokens[0] = sos_tokens[count]
        #     count+=1
        #a_indices = torch.cat((sos_tokens, a_indices), dim=1)

        logits = self.transformer(a_indices)  #transformer predict the probability of tokens
        
        #raise Exception('TODO2 step1-3!')
        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    # def inpainting(self, z_indices, mask_bc, ratio):
    #     # raise Exception('TODO3 step1-1!')
    #     a_indices = z_indices.clone()
    #     a_indices[mask_bc] = self.mask_token_id
    #     logits = self.transformer(a_indices)
    #     masked_indices = torch.ones_like(z_indices, device=z_indices.device) * self.mask_token_id
    #     mask_z_indices = mask_bc * self.mask_token_id + (~mask_bc) * z_indices # mask_bc true means mask
    #     #logits = self.transformer(mask_z_indices)
    #     mask_num = mask_bc.sum() #total number of mask token
    #     #Apply softmax to convert logits into a probability distribution across the last dimension.
    #     logits = nn.functional.softmax(logits, dim=-1)
    #     #FIND MAX probability for each token value
    #     z_indices_predict_prob, z_indices_predict = torch.max(logits, dim=-1)

    #     # ratio=None 
    #     #predicted probabilities add temperature annealing gumbel noise as confidence
    #     g = -torch.log(-torch.log(torch.rand(z_indices_predict_prob.shape))).to(z_indices.device) # gumbel noise (-log(-log(Uniform(0, 1))))
    #     temperature = self.choice_temperature * (1 - ratio)
    #     confidence = z_indices_predict_prob + temperature * g

    #     #hint: If mask is False(not masked original image token), the probability should be set to infinity, 
    #     #so that the tokens are not affected by the transformer's prediction
    #     #sort the confidence for the rank 
    #     #define how much the iteration remain predicted tokens by mask scheduling
    #     #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens
    #     # mask_bc=None

    #     z_indices_predict_prob[mask_bc == False] = float('inf')
    #     r = math.ceil((ratio) * mask_num)
    #     #print(ratio)

    #     #sorted_confidence, sorted_indices = torch.sort(confidence, descending=True)
    #     confidence[mask_bc == False] = -float('inf')
    #     non_mask_id = confidence.topk(r, dim=1).indices
    #     mask_bc.scatter_(dim=1, index=non_mask_id, value=False)

    #     final_a_indices = mask_bc * z_indices_predict + (~mask_bc) * z_indices
        
    #     return final_a_indices, mask_bc

    def inpainting(self, z_indices, mask, mask_num, ratio,pre_ratio):
        mask_b = copy.deepcopy(mask)
        masked_indices = torch.ones_like(z_indices, device=z_indices.device) * self.mask_token_id
        mask_z_indices = mask * self.mask_token_id + (~mask) * z_indices # mask_bc true means mask
        logits = self.transformer(mask_z_indices)
        # a_indices = z_indices.clone()
        # a_indices[mask] = self.mask_token_id
        # logits = self.transformer(a_indices)
        probs = F.softmax(logits, dim=-1)

        z_indices_predict_prob, z_indices_predict = torch.max(probs, dim=-1)
        g = -torch.log(-torch.log(torch.rand_like(z_indices_predict_prob)))  # gumbel noise
        temperature = self.choice_temperature * (ratio)
        confidence = z_indices_predict_prob + temperature * g

        #confidence = torch.where(mask, confidence, torch.full_like(confidence, float('inf')))

        # sorted_confidence, sorted_indices = torch.sort(confidence, descending=True)
        # mask_fill_count = math.ceil(self.gamma(ratio) * mask_num)

        # new_mask = torch.zeros_like(mask, device=mask.device)
        # new_mask.scatter_(dim=1, index=sorted_indices[:, :mask_fill_count], value=True)

        # print()
        # print(mask_num)
        # print(ratio)
        # print(ratio * mask_num)
        # print()
        # print(ratio)
        # print(pre_ratio)
        r = math.ceil(ratio * mask_num - (pre_ratio) * mask_num)
        #print(r)
        confidence[mask == False] = -float('inf')
        non_mask_id = confidence.topk(r, dim=1,largest = True).indices
        mask.scatter_(dim=1, index=non_mask_id, value=False)

        #z_indices_predict = torch.clamp(z_indices_predict, 0, self.mask_token_id - 1)
        #final_a_indices = new_mask * z_indices_predict + (~new_mask) * z_indices
        final_a_indices = mask_b * z_indices_predict + (~mask_b) * z_indices

        return z_indices_predict, mask
        #return final_a_indices, mask
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
