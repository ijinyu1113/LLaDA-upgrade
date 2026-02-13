import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from datasets import load_dataset
import matplotlib.pyplot as plt
import os

# --- 1. CORE ARCHITECTURE ---
class AMIPRouter(torch.nn.Module):
    def __init__(self, d_model=4096, K=8):
        super().__init__()
        self.routing_net = torch.nn.Linear(d_model, K)
        self.mapping_nets = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(d_model, d_model // 4),
                torch.nn.GELU(),
                torch.nn.Linear(d_model // 4, d_model)
            ) for _ in range(K)
        ])
        
    def forward(self, h_L, mask_indices, unmasked_indices, range_r=8):
        delta_h = torch.zeros_like(h_L) 
        bsz, seq_len, d_model = h_L.shape
        for b in range(bsz):
            m_idx, u_idx = mask_indices[b], unmasked_indices[b]
            for a in m_idx:
                adj = [t for t in u_idx if 0 < abs(t - a) <= range_r]
                if not adj: continue
                h_a, h_ts = h_L[b, a:a+1, :], h_L[b, adj, :] 
                weights = F.softmax(self.routing_net(h_a), dim=-1)
                expert_outputs = sum(weights[:, i:i+1] * expert(h_ts).mean(dim=0, keepdim=True) 
                                     for i, expert in enumerate(self.mapping_nets))
                delta_h[b, a, :] = F.layer_norm(expert_outputs, (d_model,))
        return delta_h

class ALALLaDA(torch.nn.Module):
    def __init__(self, base_model, d_model=4096, K=8, alpha=0.08):
        super().__init__()
        self.base_model = base_model
        self.router = AMIPRouter(d_model=d_model, K=K)
        self.alpha = alpha

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        h_L = outputs.hidden_states[-1].to(next(self.router.parameters()).dtype)
        m_idx = [torch.where(row == 126336)[0] for row in input_ids]
        u_idx = [torch.where(row != 126336)[0] for row in input_ids]
        
        refinement_delta = self.router(h_L, m_idx, u_idx)
        blended_h = h_L + (self.alpha * refinement_delta)
        
        # Recursive finder for LLaDA internal structure
        final_norm, lm_head = None, None
        for name, module in self.base_model.named_modules():
            if any(x in name.lower() for x in ["norm", "ln_f"]): final_norm = module
            if any(x in name.lower() for x in ["lm_head", "ff_out"]): lm_head = module
        
        logits = lm_head(final_norm(blended_h)) if final_norm else torch.matmul(blended_h, self.base_model.get_input_embeddings().weight.T)
        return type('Obj', (object,), {'logits': logits})

# --- 2. PERPLEXITY CALCULATOR ---
def calculate_perplexity(model, tokenizer, dataset, num_samples=10):
    encodings = tokenizer("\n\n".join(dataset["text"][:num_samples]), return_tensors="pt")
    max_length = 512
    stride = 256
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(next(model.parameters()).device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            nlls.append(loss * trg_len)
        prev_end_loc = end_loc
        if end_loc == seq_len: break

    return torch.exp(torch.stack(nlls).sum() / end_loc).item()

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    model_id = 'GSAI-ML/LLaDA-8B-Instruct'
    weights_path = "amip_router_final.pt"
    
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base = AutoModel.from_pretrained(model_id, trust_remote_code=True, quantization_config=quant_config, device_map="auto")
    
    model = ALALLaDA(base).to(torch.bfloat16)
    if os.path.exists(weights_path):
        model.router.load_state_dict(torch.load(weights_path, map_location=next(base.parameters()).device))
    
    # Load Wikitext once to save time
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    alphas = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]
    ppl_scores = []

    print("\nStarting Alpha Sensitivity Sweep...")
    for a in alphas:
        model.alpha = a
        ppl = calculate_perplexity(model, tokenizer, test_dataset)
        ppl_scores.append(ppl)
        print(f"Alpha: {a:<5} | PPL: {ppl:.2f}")

    # Plotting the "Safety Curve"
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, ppl_scores, marker='o', color='crimson', linewidth=2)
    plt.axhline(y=ppl_scores[0], color='gray', linestyle='--', label='Baseline PPL')
    plt.title("Perplexity vs. Alpha (Latent Intervention Sensitivity)")
    plt.xlabel("Alpha (Nudge Intensity)")
    plt.ylabel("Perplexity (WikiText-2)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("ppl_sensitivity_curve.png")
    print("\n[Success] Sensitivity curve saved to ppl_sensitivity_curve.png")