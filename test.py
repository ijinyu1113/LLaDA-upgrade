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
# --- 2. NEW ACCURACY METRIC: LOG-PROB CONFIDENCE ---
@torch.no_grad()
def evaluate_soft_accuracy(model, tokenizer, prompt, expected_token):
    device = model.device
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    target_id = tokenizer.encode(expected_token, add_special_tokens=False)[-1]
    
    seq = torch.cat([ids, torch.tensor([[126336]], device=device)], dim=1)
    
    # Baseline: use the helper that goes through ln_f -> ff_out
    b_logits = model.base_logits(seq)[0, -1, :]
    
    # Router
    r_logits = model(seq).logits[0, -1, :]
    
    b_prob = F.softmax(b_logits.float(), dim=-1)[target_id].item()
    r_prob = F.softmax(r_logits.float(), dim=-1)[target_id].item()
    
    return {"baseline_p": b_prob, "router_p": r_prob, "gain": r_prob - b_prob}

def plot_tradeoff_curve(results):
    temps = [r["Temp"] for r in results]
    base_jaccards = [r["BaseJaccard"] for r in results]
    router_jaccards = [r["RouterJaccard"] for r in results]
    router_accs = [r["RouterAcc"] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('Diversity (1 - Jaccard)', color='tab:blue')
    ax1.plot(temps, [1-j for j in base_jaccards], color='tab:cyan', marker='o', label='Baseline Diversity')
    ax1.plot(temps, [1-j for j in router_jaccards], color='tab:blue', marker='o', label='Router Diversity')
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Masked Prediction Accuracy', color='tab:red')
    ax2.plot(temps, router_accs, color='tab:red', marker='s', linestyle='--', label='Router Accuracy')
    ax2.legend(loc='upper right')
    
    plt.title("Diversity vs. Accuracy Tradeoff (Alpha=0.1)")
    plt.savefig("accuracy_diversity_tradeoff.png")
    print("\n[Success] Tradeoff curve saved to accuracy_diversity_tradeoff.png")
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
            logits = model(x).logits if use_router else model.base_logits(x)
            
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
        b_logits = model.base_logits(seq)[0, mask_locs[0]]

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
        out = model(seq).logits if use_router else model.base_logits(seq)
        
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
from datasets import load_dataset
@torch.no_grad()
def calculate_masked_accuracy(model, tokenizer, dataset_name="wikitext", split="test", num_samples=20, p_mask=0.15):
    print(f"\n[Accuracy Eval] Calculating Masked Prediction Accuracy...")
    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split)
    texts = [t for t in dataset["text"][:200] if len(t) > 50][:num_samples]
    
    correct_base, correct_router, total = 0, 0, 0
    
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        ids = enc["input_ids"].to(model.device)
        original = ids.clone()
        
        # Mask 15% of tokens
        mask_prob = torch.full(ids.shape, p_mask, device=ids.device)
        mask_indices = torch.bernoulli(mask_prob).bool()
        mask_indices[:, 0] = False  # Don't mask BOS
        
        masked_ids = ids.clone()
        masked_ids[mask_indices] = 126336
        
        if not mask_indices.any():
            continue
        
        # Base prediction
        b_logits = model.base_logits(masked_ids)
        b_preds = b_logits.argmax(dim=-1)
        
        # Router prediction
        r_logits = model(masked_ids).logits
        r_preds = r_logits.argmax(dim=-1)
        
        targets = original[mask_indices]
        correct_base += (b_preds[mask_indices] == targets).sum().item()
        correct_router += (r_preds[mask_indices] == targets).sum().item()
        total += targets.numel()
    
    print(f"  Baseline Mask Acc: {correct_base/total:.4f}")
    print(f"  Router Mask Acc:   {correct_router/total:.4f}")
    return correct_base/total, correct_router/total

if __name__ == "__main__":
    from transformers import BitsAndBytesConfig
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    model_id = 'GSAI-ML/LLaDA-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    base_model  = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,

        #quantization_config=quant_config,
        device_map="auto"
    )

    for name, child in base_model.named_children():
        print(f"{name:30s} -> {type(child).__name__}")
    print("\n--- Second level under 'model' ---")
    for name, child in base_model.model.named_children():
        print(f"model.{name:30s} -> {type(child).__name__}")
    print("\n--- Second level under 'model.transformer' ---")
    for name, child in base_model.model.transformer.named_children():
        print(f"model.transformer.{name:30s} -> {type(child).__name__}")

    # Initialize with alpha=0.3 to match training
    model = ALALLaDA(base_model, alpha=0.1).to(torch.bfloat16)
    device = next(base_model.parameters()).device
    model.router.to(device)

    weights_path = "amip_router_conditioned.pt" 
    if os.path.exists(weights_path):
        model.router.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Success: Conditioned Router loaded at {device}")
    else:
        print("WARNING: Using RANDOM initialization.")

    model.eval()

    # --- EVALUATION SUITE ---
    test_cases = [
        ("Triple Swap", "Alice has an apple, Bob has a banana, and Charlie has a cherry. Alice swaps with Bob. Then Bob swaps with Charlie. Now, Alice has the", "banana"),
        ("Distractor", "A gold coin is in the red box. A silver coin is in the blue bag. I replace the gold coin with a copper coin. The red box now has the", "copper"),
        ("Relational", "The mountain is taller than the hill. The building is shorter than the hill. The shortest object is the", "building"),
        ("State Swap", "I have a box and a bag. The ball is in the box. The key is in the bag. I swap them. The bag now has the", "ball")
    ]

    # Sweep temps: 0.0 gives a clean baseline vs router comparison
    sweep_configs = [0.0, 0.15, 0.3]
    tradeoff_results = []

    for temp_val in sweep_configs:
        print(f"\n" + "="*100)
        print(f"RUNNING SWEEP: Alpha=0.1 (fixed) | Temp={temp_val}")
        print("="*100)
        
        # A. Masked Prediction Accuracy (temperature-independent)
        # Only run once since masking accuracy doesn't depend on generation temp
        if temp_val == sweep_configs[0]:
            base_acc, router_acc = calculate_masked_accuracy(model, tokenizer, num_samples=20)
            cached_base_acc, cached_router_acc = base_acc, router_acc
        else:
            base_acc, router_acc = cached_base_acc, cached_router_acc
            print(f"  [Cached] Baseline Mask Acc: {base_acc:.4f} | Router Mask Acc: {router_acc:.4f}")
        
        # B. Soft Accuracy (also temperature-independent — single mask token, greedy)
        if temp_val == sweep_configs[0]:
            soft_accs_data = []
            for cat, prompt, expected in test_cases:
                acc_data = evaluate_soft_accuracy(model, tokenizer, prompt, expected)
                soft_accs_data.append(acc_data)
                print(f"  [{cat}] Base P={acc_data['baseline_p']:.4f} | Router P={acc_data['router_p']:.4f} | Gain={acc_data['gain']:+.4f}")
            avg_soft_acc = np.mean([d["router_p"] for d in soft_accs_data])
            cached_soft_acc = avg_soft_acc
        else:
            avg_soft_acc = cached_soft_acc
        
        # C. Logical Success — both baseline and router use SAME temp
        print(f"\n{'CATEGORY':<15} | {'EXPECTED':<10} | {'BASELINE':<30} | {'AMIP ROUTER':<30}")
        print("-" * 100)
        for category, prompt, expected in test_cases:
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            b_out = generate_stable(model, ids, use_router=False, temp=temp_val)
            b_ans = tokenizer.decode(b_out[0, ids.shape[1]:], skip_special_tokens=True).strip().lower()
            
            r_out = generate_stable(model, ids, use_router=True, temp=temp_val)
            r_ans = tokenizer.decode(r_out[0, ids.shape[1]:], skip_special_tokens=True).strip().lower()
            print(f"{category:<15} | {expected:<10} | {b_ans[:30]:<30} | {r_ans[:30]:<30}")

        # D. Diversity Measure — both modes use same temp
        diversity_samples = test_sample_diversity(model, tokenizer, num_samples=5, shared_temp=temp_val)
        
        # Compute Jaccard for both modes
        jaccard_results = {}
        for mode in ["Baseline", "Router"]:
            words = [s.lower().split() for s in diversity_samples[mode]]
            sims = []
            for i in range(len(words)):
                for j in range(i+1, len(words)):
                    s1, s2 = set(words[i]), set(words[j])
                    if s1 and s2: sims.append(len(s1 & s2) / len(s1 | s2))
            jaccard_results[mode] = np.mean(sims) if sims else 1.0
        
        print(f"  Baseline Jaccard: {jaccard_results['Baseline']:.4f} | Router Jaccard: {jaccard_results['Router']:.4f}")

        # E. Efficiency Index
        efficiency = (1 - jaccard_results["Router"]) * router_acc if router_acc > 0 else 0.0
        
        tradeoff_results.append({
            "Temp": temp_val, 
            "BaseAcc": base_acc, 
            "RouterAcc": router_acc,
            "BaseJaccard": jaccard_results["Baseline"],
            "RouterJaccard": jaccard_results["Router"],
            "SoftAcc": avg_soft_acc, 
            "Efficiency": efficiency
        })

    # --- FINAL SCIENTIFIC SUMMARY ---
    print("\n" + "="*130)
    print("RESEARCH SUMMARY: DIVERSITY vs. ACCURACY TRADEOFF (Alpha=0.3 fixed)")
    print("="*130)
    print(f"{'Temp':<8} | {'Base MaskAcc':<14} | {'Router MaskAcc':<16} | {'SoftAcc':<10} | {'Base Jaccard':<14} | {'Router Jaccard':<16} | {'Eff. Index':<10}")
    print("-" * 130)
    for r in tradeoff_results:
        print(f"{r['Temp']:<8} | {r['BaseAcc']:<14.4f} | {r['RouterAcc']:<16.4f} | {r['SoftAcc']:<10.4f} | {r['BaseJaccard']:<14.4f} | {r['RouterJaccard']:<16.4f} | {r['Efficiency']:<10.4f}")
    
    plot_tradeoff_curve(tradeoff_results)
    test_distribution_flatness(model, tokenizer)
    
    print("\n[Sweep Complete] Evaluation results and Pareto plots saved.")