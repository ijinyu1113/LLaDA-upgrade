import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. ARCHITECTURE: AMIP ROUTER (copied from your reference) ---
class AMIPRouter(torch.nn.Module):
    def __init__(self, d_model=4096, K=8):
        super().__init__()
        self.routing_net = torch.nn.Linear(d_model, K)
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
                h_mask = h_L[b, a:a+1, :]
                h_anchors = h_L[b, adj, :]
                weights = F.softmax(self.routing_net(h_mask), dim=-1)
                h_anchor_avg = h_anchors.mean(dim=0, keepdim=True)
                conditioned_in = torch.cat([h_anchor_avg, h_mask], dim=-1)
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
        normed = self.base_model.model.transformer.ln_f(blended_h)
        logits = self.base_model.model.transformer.ff_out(normed)
        return type('Obj', (object,), {'logits': logits})
    def base_logits(self, input_ids):
        out = self.base_model(input_ids, output_hidden_states=True)
        h = out.hidden_states[-1].to(torch.bfloat16)
        normed = self.base_model.model.transformer.ln_f(h)
        return self.base_model.model.transformer.ff_out(normed)


# --- 2. ENTROPY MEASUREMENT ---
# LLaDA is a masked diffusion model, so we measure entropy over masked positions
# during iterative unmasking, NOT autoregressive next-token prediction.

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


@torch.no_grad()
def measure_generation_entropy(
    model,
    tokenizer,
    prompt_text="One day, a cat named Whiskers found a magical portal in the library.",
    gen_length=128,
    block_length=32,
    steps=128,
    alpha=0.1,
    num_samples=5,
    device="cuda",
):
    """
    Measures per-step average entropy over all masked positions during LLaDA's
    iterative unmasking process, for both baseline and router.
    
    Each 'step' is one denoising iteration. At each step we measure the average
    entropy of the logit distribution across all currently-masked positions.
    """

    mask_id = 126336
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    total_steps = num_blocks * steps_per_block

    all_base_entropies = []
    all_router_entropies = []

    model.eval()

    for sample_idx in range(num_samples):
        print(f"  [Entropy Sample {sample_idx + 1}/{num_samples}]")

        base_entropies = []
        router_entropies = []

        # ============ BASELINE GENERATION ============
        x = torch.full((1, prompt_ids.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
        x[:, :prompt_ids.shape[1]] = prompt_ids.clone()

        for b_idx in range(num_blocks):
            b_start = prompt_ids.shape[1] + (b_idx * block_length)
            b_end = b_start + block_length

            block_mask = (x[:, b_start:b_end] == mask_id)
            transfer_schedule = get_num_transfer_tokens(block_mask, steps_per_block)

            for i in range(steps_per_block):
                mask_index = (x == mask_id)
                logits = model.base_logits(x)

                # Suppress EOS
                logits[:, :, 126081] = -torch.inf

                # Measure entropy over ALL currently masked positions
                masked_logits = logits[mask_index]  # [num_masked, vocab_size]
                if masked_logits.shape[0] > 0:
                    probs = F.softmax(masked_logits.float(), dim=-1)
                    per_token_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                    avg_entropy = per_token_entropy.mean().item()
                else:
                    avg_entropy = 0.0
                base_entropies.append(avg_entropy)

                # Decode step (greedy for determinism)
                x0 = torch.argmax(logits, dim=-1)
                probs_conf = F.softmax(logits, dim=-1)
                x0_p = torch.gather(probs_conf, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                x0_p[:, b_end:] = -float('inf')
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -float('inf'))

                transfer_idx = torch.zeros_like(x, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    _, sel_idx = torch.topk(confidence[j], k=transfer_schedule[j, i])
                    transfer_idx[j, sel_idx] = True
                x[transfer_idx] = x0[transfer_idx]

        # ============ ROUTER GENERATION ============
        x = torch.full((1, prompt_ids.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
        x[:, :prompt_ids.shape[1]] = prompt_ids.clone()

        for b_idx in range(num_blocks):
            b_start = prompt_ids.shape[1] + (b_idx * block_length)
            b_end = b_start + block_length

            block_mask = (x[:, b_start:b_end] == mask_id)
            transfer_schedule = get_num_transfer_tokens(block_mask, steps_per_block)

            for i in range(steps_per_block):
                mask_index = (x == mask_id)
                logits = model(x).logits

                # Suppress EOS
                logits[:, :, 126081] = -torch.inf

                # Measure entropy over ALL currently masked positions
                masked_logits = logits[mask_index]  # [num_masked, vocab_size]
                if masked_logits.shape[0] > 0:
                    probs = F.softmax(masked_logits.float(), dim=-1)
                    per_token_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                    avg_entropy = per_token_entropy.mean().item()
                else:
                    avg_entropy = 0.0
                router_entropies.append(avg_entropy)

                # Decode step (greedy for determinism)
                x0 = torch.argmax(logits, dim=-1)
                probs_conf = F.softmax(logits, dim=-1)
                x0_p = torch.gather(probs_conf, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                x0_p[:, b_end:] = -float('inf')
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -float('inf'))

                transfer_idx = torch.zeros_like(x, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    _, sel_idx = torch.topk(confidence[j], k=transfer_schedule[j, i])
                    transfer_idx[j, sel_idx] = True
                x[transfer_idx] = x0[transfer_idx]

        # Pad to same length if needed
        min_len = min(len(base_entropies), len(router_entropies))
        base_entropies = base_entropies[:min_len]
        router_entropies = router_entropies[:min_len]

        all_base_entropies.append(base_entropies)
        all_router_entropies.append(router_entropies)

    # Average across samples
    avg_base = np.mean(all_base_entropies, axis=0)
    avg_router = np.mean(all_router_entropies, axis=0)
    std_base = np.std(all_base_entropies, axis=0)
    std_router = np.std(all_router_entropies, axis=0)

    return avg_base, avg_router, std_base, std_router


def plot_entropy_curve(avg_base, avg_router, std_base, std_router, alpha=0.1, save_path="entropy_curve.png"):
    """
    Plots per-step entropy for baseline vs router with confidence bands.
    """
    positions = np.arange(len(avg_base))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    # ---- Top: entropy curves ----
    ax1 = axes[0]
    ax1.plot(positions, avg_base, color="steelblue", linewidth=1.5, label="Baseline", alpha=0.9)
    ax1.fill_between(positions, avg_base - std_base, avg_base + std_base, color="steelblue", alpha=0.15)

    ax1.plot(positions, avg_router, color="coral", linewidth=1.5, label=f"Router (α={alpha})", alpha=0.9)
    ax1.fill_between(positions, avg_router - std_router, avg_router + std_router, color="coral", alpha=0.15)

    ax1.set_xlabel("Denoising Step", fontsize=12)
    ax1.set_ylabel("Avg Entropy over Masked Positions (nats)", fontsize=12)
    ax1.set_title("Per-Step Token Entropy During LLaDA Unmasking: Baseline vs Router", fontsize=14)
    ax1.legend(fontsize=11, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ---- Bottom: entropy difference ----
    ax2 = axes[1]
    diff = avg_router - avg_base
    colors = ["coral" if d > 0 else "steelblue" for d in diff]
    ax2.bar(positions, diff, color=colors, alpha=0.7, width=1.0)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xlabel("Denoising Step", fontsize=12)
    ax2.set_ylabel("Δ Entropy\n(Router - Base)", fontsize=10)
    ax2.set_title("Entropy Difference Per Step", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Summary stats
    mean_diff = np.mean(diff)
    pct_higher = np.mean(diff > 0) * 100
    fig.text(
        0.5, 0.01,
        f"Mean Δ Entropy: {mean_diff:.4f} nats | Router higher at {pct_higher:.1f}% of steps",
        ha="center", fontsize=11, style="italic",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[Success] Entropy curve saved to {save_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"ENTROPY SUMMARY")
    print(f"{'='*60}")
    print(f"  Mean Baseline Entropy:  {np.mean(avg_base):.4f} nats")
    print(f"  Mean Router Entropy:    {np.mean(avg_router):.4f} nats")
    print(f"  Mean Δ Entropy:         {mean_diff:.4f} nats")
    print(f"  Router higher at:       {pct_higher:.1f}% of steps")
    print(f"  Max Baseline Entropy:   {np.max(avg_base):.4f} at step {np.argmax(avg_base)}")
    print(f"  Max Router Entropy:     {np.max(avg_router):.4f} at step {np.argmax(avg_router)}")
    print(f"{'='*60}")

    return diff


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("ENTROPY CURVE ANALYSIS")
    print("=" * 60)

    # --- Load model and router (same as your test.py) ---
    model_id = 'GSAI-ML/LLaDA-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    model = ALALLaDA(base_model, alpha=0.1).to(torch.bfloat16)
    device = next(base_model.parameters()).device
    model.router.to(device)

    weights_path = "amip_router_conditioned.pt"
    if os.path.exists(weights_path):
        model.router.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"[Success] Conditioned Router loaded at {device}")
    else:
        print("[WARNING] Using RANDOM initialization.")

    model.eval()

    # --- Run entropy measurement ---
    alpha = 0.1

    avg_base, avg_router, std_base, std_router = measure_generation_entropy(
        model=model,
        tokenizer=tokenizer,
        prompt_text="One day, a cat named Whiskers found a magical portal in the library.",
        gen_length=128,
        block_length=32,
        steps=128,
        alpha=alpha,
        num_samples=5,
        device=device,
    )

    diff = plot_entropy_curve(
        avg_base, avg_router, std_base, std_router,
        alpha=alpha,
        save_path="entropy_curve.png",
    )