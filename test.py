import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os
import matplotlib.pyplot as plt

# --- 1. ARCHITECTURE: AMIP ROUTER ---
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
                adjacent_items = [t for t in u_idx if 0 < abs(t - a) <= range_r]
                if not adjacent_items: continue
                h_a, h_ts = h_L[b, a:a+1, :], h_L[b, adjacent_items, :] 
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

# --- 2. MATH UTILS ---
def calculate_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

def calculate_jsd(p_logits, q_logits):
    p, q = F.softmax(p_logits, dim=-1), F.softmax(q_logits, dim=-1)
    m = 0.5 * (p + q)
    jsd = 0.5 * (F.kl_div(torch.log(p + 1e-9), m, reduction='batchmean') + 
                 F.kl_div(torch.log(q + 1e-9), m, reduction='batchmean'))
    return jsd.item()

def add_gumbel_noise(logits, temperature):
    if temperature == 0: return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    return logits.exp() / ((- torch.log(noise)) ** temperature)

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base, remainder = mask_num // steps, mask_num % steps
    res = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)): res[i, :remainder[i]] += 1
    return res

# --- 3. GENERATION & ANALYSIS ---
# --- Updated Generation Logic based on official LLaDA implementation ---
@torch.no_grad()
def generate_stable(model, prompt_ids, steps=128, gen_length=128, block_length=32, use_router=True, temp=0.0):
    device, mask_id = model.device, 126336
    x = torch.full((prompt_ids.shape[0], prompt_ids.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
    x[:, :prompt_ids.shape[1]] = prompt_ids.clone()
    
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    for b in range(num_blocks):
        # Determine the boundaries of the current active block
        b_start = prompt_ids.shape[1] + (b * block_length)
        b_end = b_start + block_length
        
        # Calculate tokens to unmask ONLY for this block (Official LLaDA Step)
        block_mask = (x[:, b_start:b_end] == mask_id)
        transfer_schedule = get_num_transfer_tokens(block_mask, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            logits = model(x).logits if use_router else model.base_model(x).logits
            
            # Official Heuristic: Prevent premature stopping
            logits[:, :, 126081] = -torch.inf 
            
            # Official Gumbel Sampling
            logits_noise = add_gumbel_noise(logits, temperature=temp)
            x0 = torch.argmax(logits_noise, dim=-1)
            
            # Official Confidence Scoring
            probs = F.softmax(logits, dim=-1)
            x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            
            # Official Block-Masking: Ignore future tokens
            x0_p[:, b_end:] = -np.inf
            
            # Official Transfer Logic
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            transfer_idx = torch.zeros_like(x, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                _, sel_idx = torch.topk(confidence[j], k=transfer_schedule[j, i])
                transfer_idx[j, sel_idx] = True
            
            x[transfer_idx] = x0[transfer_idx]
            
    return x

@torch.no_grad()
def run_scientific_eval(model, tokenizer, prompt):
    device = model.device
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    results = {"gap": [], "jsd": [], "steps": []}
    
    # We use a single sequence to compare distributions at the exact same latent state
    seq = torch.cat([ids, torch.full((1, 32), 126336, device=device)], dim=1)
    
    for t in range(32):
        mask_locs = (seq == 126336).nonzero(as_tuple=True)[1]
        if not len(mask_locs): break
        
        # Get simultaneous distributions
        b_logits = model.base_model(seq).logits[0, mask_locs[0]]
        r_logits = model(seq).logits[0, mask_locs[0]]
        
        # 1. Calculate JSD (The real divergence)
        results["jsd"].append(calculate_jsd(b_logits, r_logits))
        
        # 2. Calculate Entropy Gap
        b_h = calculate_entropy(F.softmax(b_logits, dim=-1))
        r_h = calculate_entropy(F.softmax(r_logits, dim=-1))
        results["gap"].append((r_h - b_h).item())
        
        # 3. Step forward using the Router's choice to see logic flow
        _, pred = F.softmax(r_logits, dim=-1).max(dim=-1)
        seq[0, mask_locs[0]] = pred
        results["steps"].append(t)
            
    return results

@torch.no_grad()
def test_distribution_flatness(model, tokenizer):
    prompt = "Choose a number between 1 and 10:"
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    target_tokens = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 11)]
    
    print(f"\n[Flatness Test] Prompt: '{prompt}'")
    results = {}
    for mode in ["Baseline", "Router"]:
        use_router = (mode == "Router")
        seq = torch.cat([ids, torch.full((1, 1), 126336, device=model.device)], dim=1)
        out = model(seq).logits if use_router else model.base_model(seq).logits
        
        probs = F.softmax(out[0, -1, :], dim=-1)
        number_probs = probs[target_tokens]
        
        # Calculate Flatness (Normalized Entropy)
        flatness = -torch.sum(number_probs * torch.log(number_probs + 1e-9)).item()
        
        # FIX: Cast to float32 before numpy conversion
        results[mode] = (number_probs.detach().to(torch.float32).cpu().numpy(), flatness)
        print(f"{mode} Entropy (Flatness): {flatness:.4f}")

    return results

@torch.no_grad()
def test_sample_diversity(model, tokenizer, num_samples=3, shared_temp=0.1):
    prompt = "Write a short story about a cat who finds a magical portal in a library."
    filename = "story_samples_diversity.txt"
    
    print(f"\n" + "="*50)
    print(f"DIVERSITY TEST: Sampling {num_samples} times | Temp={shared_temp}")
    print(f"Saving outputs to: {filename}")
    print("="*50)
    
    all_outputs = {"Baseline": [], "Router": []}
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"PROMPT: {prompt}\n")
        f.write(f"TEMPERATURE: {shared_temp}\n")
        f.write("="*50 + "\n\n")

        for mode in ["Baseline", "Router"]:
            f.write(f"--- MODE: {mode} ---\n")
            print(f"\n>>> Generating {mode} Samples:")
            
            for s in range(num_samples):
                ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
                out = generate_stable(model, ids, steps=64, gen_length=64, use_router=(mode=="Router"), temp=shared_temp)
                text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
                
                all_outputs[mode].append(text)
                
                # Print to terminal
                print(f"  [Sample {s+1}]: {text[:150]}...") 
                
                # Write to file
                f.write(f"Sample {s+1}:\n{text}\n\n")
            
            f.write("-" * 30 + "\n")

    # Calculate Metrics
    print("\n" + "="*50)
    print("DIVERSITY METRICS SUMMARY")
    print("="*50)
    
    for mode, stories in all_outputs.items():
        words_list = [s.lower().split() for s in stories]
        flat_words = [item for sublist in words_list for item in sublist]
        
        # 1. Unique Token Ratio (Lexical Richness)
        unique_ratio = len(set(flat_words)) / len(flat_words) if flat_words else 0
        
        # 2. Pairwise Jaccard Similarity (Lower = Better Diversity)
        # Measures the overlap between samples. 1.0 means samples are identical.
        similarities = []
        for i in range(len(words_list)):
            for j in range(i + 1, len(words_list)):
                s1, s2 = set(words_list[i]), set(words_list[j])
                if not s1 or not s2: continue
                similarities.append(len(s1 & s2) / len(s1 | s2))
        
        avg_jaccard = np.mean(similarities) if similarities else 1.0
        
        print(f"[{mode}] Unique Token Ratio: {unique_ratio:.4f}")
        print(f"[{mode}] Avg Jaccard Similarity: {avg_jaccard:.4f}")

    return all_outputs

if __name__ == "__main__":
    model_id = 'GSAI-ML/LLaDA-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16).to('cuda')
    model = ALALLaDA(base_model).to('cuda').to(torch.bfloat16)

    # 2. LOAD YOUR TRAINED WEIGHTS (Fixes the "Random" issue)
    # Replace "trained_router.pt" with the actual path to your saved file
    weights_path = "amip_router_final.pt" 
    if os.path.exists(weights_path):
        model.router.load_state_dict(torch.load(weights_path))
        print(f"Successfully loaded trained AMIP Router from {weights_path}")
    else:
        print("WARNING: No trained weights found. Running with a RANDOM router.")

    model.eval()

    # Hard Logic Cases
    test_cases = [
        ("Triple Swap", "Alice has an apple, Bob has a banana, and Charlie has a cherry. Alice swaps with Bob. Then Bob swaps with Charlie. Now, Alice has the", "banana"),
        ("Distractor", "A gold coin is in the red box. A silver coin is in the blue bag. I replace the gold coin with a copper coin. The red box now has the", "copper"),
        ("Relational", "The mountain is taller than the hill. The building is shorter than the hill. The shortest object is the", "building"),
        ("State Swap", "I have a box and a bag. The ball is in the box. The key is in the bag. I swap them. The bag now has the", "ball")
    ]

    # Parameter Sweep Configurations
    # Format: (Alpha, Temp)
    sweep_configs = [
        (0.05, 0.1),  # Conservative: High stability, lower reasoning
        (0.08, 0.3),  # Balanced: The "Sweet Spot"
        (0.12, 0.5)   # Aggressive: High reasoning gap, higher risk of noise
    ]

    for alpha_val, temp_val in sweep_configs:
        model.alpha = alpha_val
        print(f"\n" + "="*100)
        print(f"RUNNING SWEEP: Alpha={alpha_val} | Temp={temp_val}")
        print("="*100)
        print(f"{'CATEGORY':<15} | {'EXPECTED':<10} | {'BASELINE':<30} | {'AMIP ROUTER':<30}")
        print("-" * 100)

        sweep_results = {"baseline": 0, "router": 0}

        for category, prompt, expected in test_cases:
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to('cuda')
            
            # Baseline uses fixed temp=0.1 for maximum stability
            b_out = generate_stable(model, ids, use_router=False, temp=0.0)
            b_ans = tokenizer.decode(b_out[0, ids.shape[1]:], skip_special_tokens=True).strip().lower()
            
            # Router uses the sweep temperature
            r_out = generate_stable(model, ids, use_router=True, temp=temp_val)
            r_ans = tokenizer.decode(r_out[0, ids.shape[1]:], skip_special_tokens=True).strip().lower()

            if expected in b_ans: sweep_results["baseline"] += 1
            if expected in r_ans: sweep_results["router"] += 1

            print(f"{category:<15} | {expected:<10} | {b_ans[:30]:<30} | {r_ans[:30]:<30}")

        print("-" * 100)
        print(f"CFG SUMMARY | Baseline: {sweep_results['baseline']}/{len(test_cases)} | AMIP Router: {sweep_results['router']}/{len(test_cases)}")
        
        # 2. RUN MECHANISTIC ANALYSIS ON THE HARDEST CASE for each config
        print(f"[Analysis] Capturing Latent Expansion for Alpha {alpha_val}...")
        stats = run_scientific_eval(model, tokenizer, test_cases[0][1])
        
        plt.figure(figsize=(10, 6))
        plt.plot(stats["steps"], stats["gap"], label=f'Alpha={alpha_val}')
        plt.fill_between(stats["steps"], 0, stats["gap"], alpha=0.1)
        plt.title(f"Reasoning Gap Scaling (Alpha={alpha_val})")
        plt.ylabel("Entropy Nudge")
        plt.savefig(f"analysis_alpha_{alpha_val}.png")
        plt.close() # Close to prevent overlapping plots
    flatness_data = test_distribution_flatness(model, tokenizer)
    story_samples = test_sample_diversity(model, tokenizer, num_samples=5)
    
    # Plotting the flatness
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, 11), flatness_data["Baseline"][0], alpha=0.5, label='Baseline')
    plt.bar(range(1, 11), flatness_data["Router"][0], alpha=0.5, label='AMIP Router')
    plt.title("Flatness Test: Probability of choosing numbers 1-10")
    plt.legend()
    plt.savefig("flatness_test.png")
    print("\n" + "="*100)
    print("SWEEP COMPLETE: All plots saved to directory.")
    print("="*100)