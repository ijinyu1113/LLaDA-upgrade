import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# --- 1. ARCHITECTURE: AMIP ROUTER & MAPPING NETWORKS ---
# Implements the routing-mapping mechanism (Eq. 11, 12, 13) [cite: 360-372]
class AMIPRouter(torch.nn.Module):
    def __init__(self, d_model=4096, K=8):
        super().__init__()
        # Routing network g(x): Eq. 12 [cite: 365]
        self.routing_net = torch.nn.Linear(d_model, K)
        
        # Mapping networks f_i(x): Eq. 11 [cite: 361]
        self.mapping_nets = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(d_model, d_model // 4),
                torch.nn.GELU(),
                torch.nn.Linear(d_model // 4, d_model)
            ) for _ in range(K)
        ])
        
    def forward(self, item_repr, target_mask_repr):
        """
        item_repr: h_t^L (Contextualized item representation) [cite: 354]
        target_mask_repr: h_a^L (Contextualized mask representation) [cite: 358]
        """
        # Calculate routing weights: g(h_a^L) [cite: 370]
        weights = F.softmax(self.routing_net(target_mask_repr), dim=-1) # [B, K]
        
        # Refine representation via Experts: Eq. 13 [cite: 370]
        refined = 0
        for i, expert in enumerate(self.mapping_nets):
            refined += weights[:, i:i+1] * expert(item_repr)
            
        return F.layer_norm(refined, (refined.shape[-1],)) 

# --- 2. HELPER FUNCTIONS: MASKING & ADJACENCY ---
def apply_random_mask(input_ids, p_mask, mask_token_id=126336):
    """Applies dynamic masking from 0% to 100% across the sequence. [cite: 314-315]"""
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, p_mask)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # Only mask tokens, don't calculate loss on unmasked tokens
    input_ids[masked_indices] = mask_token_id
    labels[~masked_indices] = -100 # Standard cross-entropy ignore index
    return input_ids, labels

def find_adjacent_pairs(input_ids, mask_token_id=126336, range_r=2):
    """
    Finds 2nd-order adjacent pairs: 0 < |t - a| <= 2. [cite: 355, 384]
    t: Unmasked item index (m_t = 0)
    a: Masked target index (m_a = 1)
    """
    unmasked_idx = []
    masked_idx = []
    
    bsz, seq_len = input_ids.shape
    for b in range(bsz):
        for t in range(seq_len):
            if input_ids[b, t] != mask_token_id: 
                # Look for adjacent masks within +/- range_r [cite: 355]
                for offset in range(-range_r, range_r + 1):
                    a = t + offset
                    if 0 <= a < seq_len and offset != 0:
                        if input_ids[b, a] == mask_token_id: 
                            unmasked_idx.append((b, t))
                            masked_idx.append((b, a))
                            
    return unmasked_idx, masked_idx

# --- 3. LOSS FUNCTIONS ---
def calculate_amip_loss(refined_repr, target_item_ids, base_model_embeddings):
    """LAMIP: Predicts adjacent masked items using refined representations (Eq. 14). [cite: 377-381]"""
    logits = torch.matmul(refined_repr, base_model_embeddings.weight.T)
    return F.cross_entropy(logits, target_item_ids)

def calculate_reg_loss(mask_repr, target_item_ids, base_model_embeddings):
    """Lreg: Retains knowledge of Masked Item Prediction (Eq. 16). [cite: 385-390]"""
    logits = torch.matmul(mask_repr, base_model_embeddings.weight.T)
    return F.cross_entropy(logits, target_item_ids)

# --- 4. MAIN TRAINING LOOP ---
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "GSAI-ML/LLaDA-8B-Instruct"
    mask_token_id = 126336
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_llada = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    base_llada.eval()
    
    router = AMIPRouter(d_model=4096, K=8).to(device).to(torch.bfloat16)
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-4)
    
    # FIX: Pre-filter dataset to avoid empty batches and IndexError
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.filter(lambda x: len(x["text"]) > 40) # Ensure sufficient context
    
    def collate_fn(batch):
        texts = [x["text"] for x in batch]
        return tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    print("Starting AMIP-Diffusion Router Training...")
    

    for step, batch in enumerate(tqdm(loader)):
        if step > 500: break # Training pilot length
        
        input_ids = batch["input_ids"].to(device)
        
        # 1. Dynamic Masking (0% to 100%) [cite: 314-315]
        p_mask = torch.rand(1).item()
        masked_ids, labels = apply_random_mask(input_ids, p_mask, mask_token_id)
        
        # 2. Get Representations H^L [cite: 353]
        with torch.no_grad():
            outputs = base_llada(masked_ids, output_hidden_states=True)
            h_L = outputs.hidden_states[-1] 

        # 3. Identify Adjacent Pairs (t -> a) 
        u_idx, m_idx = find_adjacent_pairs(masked_ids, mask_token_id, range_r=2)
        if not u_idx: continue
        
        # Gather representations for the batch
        b_u, t_idx = zip(*u_idx)
        b_m, a_idx = zip(*m_idx)
        h_t = h_L[torch.tensor(b_u), torch.tensor(t_idx)] # Unmasked item repr
        h_a = h_L[torch.tensor(b_m), torch.tensor(a_idx)] # Masked target repr
        target_labels = labels[torch.tensor(b_m), torch.tensor(a_idx)]
        
        # 4. Refine via Router-Mapping mechanism [cite: 369-370]
        refined_h = router(h_t, h_a)
        
        # 5. Dual Objective Loss (Eq. 17) [cite: 394]
        l_amip = calculate_amip_loss(refined_h, target_labels, base_llada.get_input_embeddings())
        l_reg = calculate_reg_loss(h_a, target_labels, base_llada.get_input_embeddings())
        
        total_loss = l_amip + (0.3 * l_reg) # lambda_reg = 0.3 [cite: 394, 901]
        
        # 6. Optimize
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.save(router.state_dict(), "amip_router_final.pt")
    print("Training Complete. Weights saved to amip_router_final.pt.")

if __name__ == "__main__":
    train()