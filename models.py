import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
import os
import matplotlib.pyplot as plt

# --- 1. ARCHITECTURE: AMIP ROUTER ---
class AMIPRouter(torch.nn.Module):
    def __init__(self, d_model=4096, K=8):
        super().__init__()
        self.routing_net = torch.nn.Linear(d_model, K)
        # Experts now take [Anchor || Mask] input
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(d_model * 2, d_model // 4),
                torch.nn.GELU(),
                torch.nn.Linear(d_model // 4, d_model)
            ) for _ in range(K)
        ])
        
    def forward(self, h_L, mask_indices, unmasked_indices, range_r=5):
        delta_h = torch.zeros_like(h_L) 
        bsz, seq_len, d_model = h_L.shape
        for b in range(bsz):
            m_idx, u_idx = mask_indices[b], unmasked_indices[b]
            for a in m_idx:
                adj = [t for t in u_idx if 0 < abs(t - a) <= range_r]
                if not adj: continue
                # Conditioned Input: Expert sees Anchor AND the "Hole" (Mask)
                h_mask = h_L[b, a:a+1, :]
                h_anchors = h_L[b, adj, :]
                
                weights = F.softmax(self.routing_net(h_mask), dim=-1)
                
                # Combine Mask with the mean of its nearby Anchors
                h_anchor_avg = h_anchors.mean(dim=0, keepdim=True)
                conditioned_in = torch.cat([h_anchor_avg, h_mask], dim=-1) # [1, 8192]
                
                expert_out = sum(weights[:, i:i+1] * exp(conditioned_in) for i, exp in enumerate(self.experts))
                delta_h[b, a, :] = F.layer_norm(expert_out, (d_model,))
        return delta_h

class ALALLaDA(torch.nn.Module):
    def __init__(self, base_model, alpha=0.08):
        super().__init__()
        self.base_model = base_model
        self.router = AMIPRouter()
        self.alpha = alpha
    @property
    def device(self):
        return next(self.base_model.parameters()).device
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        h_L = outputs.hidden_states[-1].to(torch.bfloat16)
        
        m_idx = [torch.where(row == 126336)[0] for row in input_ids]
        u_idx = [torch.where(row != 126336)[0] for row in input_ids]
        
        delta = self.router(h_L, m_idx, u_idx)
        blended_h = ((1 - self.alpha) * h_L) + (self.alpha * delta)
        
        # Direct access instead of searching
        normed = self.base_model.model.transformer.ln_f(blended_h)
        logits = self.base_model.model.transformer.ff_out(normed)
        
        return type('Obj', (object,), {'logits': logits})
    def base_logits(self, input_ids):
        out = self.base_model(input_ids, output_hidden_states=True)
        h = out.hidden_states[-1].to(torch.bfloat16)
        normed = self.base_model.model.transformer.ln_f(h)
        return self.base_model.model.transformer.ff_out(normed)