import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
# --- 1. ARCHITECTURE: AMIP ROUTER & MAPPING NETWORKS ---
# Implements the routing-mapping mechanism (Eq. 11, 12, 13) [cite: 360-372]
class AMIPRouter(torch.nn.Module):
    def __init__(self, d_model=4096, K=8):
        super().__init__()
        # Router still looks at the mask to select experts
        self.routing_net = torch.nn.Linear(d_model, K)
        
        # Experts now take [Anchor || Mask] -> Input Dim = d_model * 2
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(d_model * 2, d_model // 4),
                torch.nn.GELU(),
                torch.nn.Linear(d_model // 4, d_model)
            ) for _ in range(K)
        ])
        
    def forward(self, h_anchor, h_mask):
        # 1. Selection logic based on the mask's "void"
        weights = F.softmax(self.routing_net(h_mask), dim=-1) # [B, K]
        
        # 2. Conditioned Input: Expert sees both Source and Target
        conditioned_input = torch.cat([h_anchor, h_mask], dim=-1) # [B, 8192]
        
        # 3. Weighted Expert Output
        refined_delta = 0
        for i, expert in enumerate(self.experts):
            refined_delta += weights[:, i:i+1] * expert(conditioned_input)
            
        return refined_delta # Returns the "Delta" to be interpolated

# --- 2. HELPER FUNCTIONS: MASKING & ADJACENCY ---
def apply_random_mask(input_ids,attention_mask, p_mask, mask_token_id=126336):
    """Applies dynamic masking from 0% to 100% across the sequence. [cite: 314-315]"""
    labels = input_ids.clone()
    
    # Create a mask only for "real" tokens (where attention_mask == 1)
    probability_matrix = torch.full(labels.shape, p_mask).to(input_ids.device)
    probability_matrix = probability_matrix * attention_mask # Zero out padding
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    input_ids[masked_indices] = mask_token_id
    
    # IMPORTANT: Only calculate loss on real tokens that were masked
    labels[~masked_indices] = -100 
    return input_ids, labels

def find_adjacent_pairs(input_ids, mask_token_id=126336, range_r=5):
    """
    Finds 2nd-order adjacent pairs: 0 < |t - a| <= range_r. 
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
    
def evaluate(router, base_llada, loader, device, mask_token_id, alpha=0.1):
    """Computes validation loss to track generalization."""
    router.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in loader:
            attention_mask = batch["attention_mask"].to(device)
            input_ids = batch["input_ids"].to(device)
            
            # Use a fixed p_mask for evaluation stability
            masked_ids, labels = apply_random_mask(input_ids, attention_mask, 0.15, mask_token_id)
            
            outputs = base_llada(masked_ids, output_hidden_states=True)
            h_L = outputs.hidden_states[-1].to(torch.bfloat16)

            u_idx, m_idx = find_adjacent_pairs(masked_ids, mask_token_id, range_r=5)
            if not u_idx: continue
            
            h_t = h_L[torch.tensor([x[0] for x in u_idx]), torch.tensor([x[1] for x in u_idx])]
            h_a = h_L[torch.tensor([x[0] for x in m_idx]), torch.tensor([x[1] for x in m_idx])]
            target_labels = labels[torch.tensor([x[0] for x in m_idx]), torch.tensor([x[1] for x in m_idx])]
            
            # Convex Interpolation Logic
            delta = router(h_t, h_a)
            h_blended = ((1 - alpha) * h_a) + (alpha * delta)
            
            # Matching the training-time projection exactly
            h_blended = h_blended.float()
            normed = base_llada.model.transformer.ln_f(h_blended)
            logits = base_llada.model.transformer.ff_out(normed.to(base_llada.model.transformer.ff_out.weight.dtype))
            
            loss = F.cross_entropy(logits, target_labels)
            val_loss += loss.item()
            
    router.train()
    return val_loss / len(loader) if len(loader) > 0 else 0

# --- 4. MAIN TRAINING LOOP ---
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "GSAI-ML/LLaDA-8B-Instruct"
    mask_token_id = 126336
    
    # Quantization to fit 8B on consumer GPU
    #quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_llada = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, device_map="auto"
    )
    
    router = AMIPRouter(d_model=4096, K=8).to(device).to(torch.bfloat16)
    optimizer = torch.optim.AdamW(router.parameters(), lr=2e-4) # Slightly lower LR for stability
    
    full_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_data = full_dataset["train"].filter(lambda x: len(x["text"]) > 50)
    val_data = full_dataset["validation"].filter(lambda x: len(x["text"]) > 50)
    
    def collate_fn(batch):
        return tokenizer([x["text"] for x in batch], return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=2, shuffle=False, collate_fn=collate_fn)

    best_val_loss = float('inf')
    print("Training Conditioned Residual Router with Validation...")

    for step, batch in enumerate(tqdm(train_loader)):
        # Training logic (Masking -> Representations -> Router -> Loss)
        attention_mask = batch["attention_mask"].to(device)
        input_ids = batch["input_ids"].to(device)
        p_mask = torch.rand(1).item()
        masked_ids, labels = apply_random_mask(input_ids, attention_mask, p_mask, mask_token_id)
        
        with torch.no_grad():
            h_L = base_llada(masked_ids, output_hidden_states=True).hidden_states[-1]

        u_idx, m_idx = find_adjacent_pairs(masked_ids, mask_token_id, range_r=5)
        if not u_idx: continue
        
        h_t = h_L[torch.tensor([x[0] for x in u_idx]), torch.tensor([x[1] for x in u_idx])]
        h_a = h_L[torch.tensor([x[0] for x in m_idx]), torch.tensor([x[1] for x in m_idx])]
        target_labels = labels[torch.tensor([x[0] for x in m_idx]), torch.tensor([x[1] for x in m_idx])]
        
        delta = router(h_t.to(torch.bfloat16), h_a.to(torch.bfloat16))
        
        alpha = 0.1
        h_blended = ((1 - alpha) * h_a) + (alpha * delta)
        h_blended = h_blended.float()
        normed = base_llada.model.transformer.ln_f(h_blended)
        logits = base_llada.model.transformer.ff_out(normed.to(base_llada.model.transformer.ff_out.weight.dtype))
        
        loss = F.cross_entropy(logits, target_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Validation Check
        if step % 500 == 0 and step > 0:
            val_loss = evaluate(router, base_llada, val_loader, device, mask_token_id)
            print(f"Step {step} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(router.state_dict(), "amip_router_best.pt")
                print(f"New Best Model Saved (Val Loss: {val_loss:.4f})")

if __name__ == "__main__":
    train()
