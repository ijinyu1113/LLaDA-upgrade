import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from models import AMIPRouter


def load_model_and_router(device="cuda"):
    model_name = "GSAI-ML/LLaDA-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.eval()

    d_model = base_model.config.hidden_size
    router = AMIPRouter(d_model=d_model, K=8)

    actual_device = next(base_model.parameters()).device
    router = router.to(actual_device).to(torch.bfloat16)
    router.load_state_dict(torch.load("amip_router_conditioned.pt", map_location=actual_device))
    router.eval()

    return base_model, tokenizer, router, actual_device


def get_hidden_states(model, input_ids):
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        return outputs.hidden_states[-1].to(torch.bfloat16)


def get_logits_from_hidden(model, h):
    """Pass hidden states through LLaDA's ln_f -> ff_out to get logits."""
    normed = model.model.transformer.ln_f(h)
    logits = model.model.transformer.ff_out(normed)
    return logits


def apply_router_to_hidden(router, h, mask_indices, unmasked_indices, alpha=0.3, range_r=5):
    """
    Apply router exactly as ALALLaDA does it.
    Returns blended hidden states.
    """
    delta = router(h, mask_indices, unmasked_indices, range_r=range_r)
    blended = (1 - alpha) * h + alpha * delta
    return blended


def create_test_problems():
    problems = {
        "triple_swap": {
            "text": "Alice has the banana, Bob has the cherry, and Charlie has the apple. Alice swaps with Bob. Bob swaps with Charlie. Alice swaps with Charlie. Now, Alice has the",
            "answer": "banana",
        },
        "distractor": {
            "text": "The red bag has the gold coin, and the blue bag has the copper coin. The green bag has the silver coin. I take the coin from the red bag. The red bag now has the",
            "answer": "copper",
        },
        "state_swap": {
            "text": "The basket has the ball, and the box has the key. I swap the contents of the basket and the box. The basket now has the",
            "answer": "key",
        },
        "relational": {
            "text": "The mountain is taller than the tree. The tree is taller than the building. What is the order of the objects from tallest to shortest? The tallest object is the",
            "answer": "mountain",
        },
    }
    return problems

# Replace the answer_ids logic with this:
def get_answer_token_id(tokenizer, context, answer):
    """Find the token ID the model would use for 'answer' following 'context'."""
    # The model sees " banana" not "banana" — tokenize in context
    with_answer = tokenizer(context + " " + answer, add_special_tokens=False).input_ids
    without_answer = tokenizer(context, add_special_tokens=False).input_ids
    # The answer token is the first token after the context
    answer_token_id = with_answer[len(without_answer)]
    return answer_token_id
def run_diagnostic(base_model, tokenizer, router, device):
    MASK_ID = 126336
    problems = create_test_problems()
    alphas = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

    print("=" * 100)
    print("AMIP ROUTER LATENT SPACE DIAGNOSTIC")
    print("=" * 100)

    for name, prob in problems.items():
        print(f"\n{'─' * 100}")
        print(f"Problem: {name.upper()}")
        print(f"Input: {prob['text']}")
        print(f"Expected: {prob['answer']}")
        print(f"{'─' * 100}")

        # Tokenize and append one mask token at the end (the position we want to predict)
        text_ids = tokenizer(prob["text"], return_tensors="pt")["input_ids"].to(device)
        seq_len = text_ids.shape[1]
        query_pos = seq_len  # the appended mask token position

        # Append mask token
        masked_input = torch.cat(
            [text_ids, torch.tensor([[MASK_ID]], device=device)], dim=1
        )

        # Build mask/unmasked index lists (same format as ALALLaDA)
        m_idx = [torch.where(masked_input[b] == MASK_ID)[0] for b in range(masked_input.shape[0])]
        u_idx = [torch.where(masked_input[b] != MASK_ID)[0] for b in range(masked_input.shape[0])]

        # Get base hidden states
        h_base = get_hidden_states(base_model, masked_input)
        h_query_base = h_base[0, query_pos, :]

        # Get base logits and prediction
        with torch.no_grad():
            base_logits_full = get_logits_from_hidden(base_model, h_base)
            base_logits = base_logits_full[0, query_pos, :]
            base_probs = F.softmax(base_logits.float(), dim=-1)
            base_pred_id = base_logits.argmax().item()
            base_pred_token = tokenizer.decode([base_pred_id]).strip()
            base_entropy = -(base_probs * torch.log(base_probs + 1e-10)).sum().item()

            # Probability of correct answer
            answer_token_id = get_answer_token_id(tokenizer, prob["text"], prob["answer"])
            print(f"  Answer token ID: {answer_token_id} -> '{tokenizer.decode([answer_token_id])}' (repr: {repr(tokenizer.decode([answer_token_id]))})")

            # Then everywhere you use answer_ids[0], use answer_token_id instead
            base_answer_prob = base_probs[answer_token_id].item()

        # Top 5 base predictions
        top5_vals, top5_ids = base_probs.topk(5)
        top5_tokens = [tokenizer.decode([tid]).strip() for tid in top5_ids.tolist()]
        top5_str = ", ".join([f"'{t}' ({v:.4f})" for t, v in zip(top5_tokens, top5_vals.tolist())])

        print(f"  Base prediction: '{base_pred_token}' (P={base_probs[base_pred_id].item():.4f})")
        print(f"  Base P(correct='{prob['answer']}'): {base_answer_prob:.6f}")
        print(f"  Base entropy: {base_entropy:.4f}")
        print(f"  Base h norm: {h_query_base.norm().item():.4f}")
        print(f"  Base top-5: {top5_str}")
        print()

        # Alpha sweep header
        print(
            f"  {'Alpha':<8} {'Cosine':<10} {'Delta/H':<10} {'DeltaNorm':<12} "
            f"{'Pred':<15} {'P(pred)':<10} {'P(correct)':<12} {'Entropy':<10} {'Direction'}"
        )
        print(f"  {'─' * 97}")

        for alpha in alphas:
            with torch.no_grad():
                h_blended = apply_router_to_hidden(
                    router, h_base.clone(), m_idx, u_idx, alpha=alpha, range_r=5
                )
                h_query_router = h_blended[0, query_pos, :]

                # Cosine similarity
                cos_sim = F.cosine_similarity(
                    h_query_base.float().unsqueeze(0),
                    h_query_router.float().unsqueeze(0),
                    dim=-1,
                ).item()

                # Delta analysis
                delta = h_query_router - h_query_base
                delta_norm = delta.float().norm().item()
                base_norm = h_query_base.float().norm().item()
                ratio = delta_norm / (base_norm + 1e-10)

                # Get logits from blended hidden states
                router_logits_full = get_logits_from_hidden(base_model, h_blended)
                router_logits = router_logits_full[0, query_pos, :]
                router_probs = F.softmax(router_logits.float(), dim=-1)
                router_pred_id = router_logits.argmax().item()
                router_pred_token = tokenizer.decode([router_pred_id]).strip()
                router_entropy = -(router_probs * torch.log(router_probs + 1e-10)).sum().item()

                router_answer_prob = router_probs[answer_token_id].item()

            # Direction indicator
            if router_answer_prob > base_answer_prob * 1.1:
                direction = "✓ BETTER"
            elif router_answer_prob < base_answer_prob * 0.9:
                direction = "✗ WORSE"
            else:
                direction = "~ SAME"

            print(
                f"  {alpha:<8.2f} {cos_sim:<10.6f} {ratio:<10.6f} {delta_norm:<12.4f} "
                f"{router_pred_token:<15} {router_probs[router_pred_id].item():<10.4f} "
                f"{router_answer_prob:<12.6f} {router_entropy:<10.4f} {direction}"
            )

        # After alpha sweep: show top-5 at best alpha
        # Find alpha where P(correct) is highest
        best_alpha = None
        best_p = base_answer_prob
        for alpha in alphas:
            with torch.no_grad():
                h_blended = apply_router_to_hidden(
                    router, h_base.clone(), m_idx, u_idx, alpha=alpha, range_r=5
                )
                rl = get_logits_from_hidden(base_model, h_blended)[0, query_pos, :]
                rp = F.softmax(rl.float(), dim=-1)
                ap = rp[answer_token_id].item()
                if ap > best_p:
                    best_p = ap
                    best_alpha = alpha

        if best_alpha is not None:
            with torch.no_grad():
                h_blended = apply_router_to_hidden(
                    router, h_base.clone(), m_idx, u_idx, alpha=best_alpha, range_r=5
                )
                rl = get_logits_from_hidden(base_model, h_blended)[0, query_pos, :]
                rp = F.softmax(rl.float(), dim=-1)
                top5_v, top5_i = rp.topk(5)
                top5_t = [tokenizer.decode([tid]).strip() for tid in top5_i.tolist()]
                top5_s = ", ".join([f"'{t}' ({v:.4f})" for t, v in zip(top5_t, top5_v.tolist())])
            print(f"\n  Best alpha={best_alpha}: top-5 = {top5_s}")
        else:
            print(f"\n  No alpha improved over baseline.")

    # Summary
    print(f"\n{'=' * 100}")
    print("INTERPRETATION GUIDE")
    print("=" * 100)
    print(
        """
  Cosine Sim:
    > 0.999   = Router barely changes direction (negligible effect)
    0.99-0.999 = Gentle nudge (probably healthy range)
    0.95-0.99  = Significant shift (check if direction helps)
    < 0.95     = Major displacement (likely destructive)

  Delta/H Ratio:
    < 0.01   = Negligible perturbation
    0.01-0.05 = Subtle (good for fine-grained steering)
    0.05-0.15 = Moderate (sweet spot if direction is correct)
    0.15-0.30 = Strong (risky)
    > 0.30   = Destructive

  KEY DIAGNOSTICS:
    - If P(correct) NEVER beats baseline at ANY alpha → router direction is wrong, need retraining
    - If P(correct) peaks then drops → sweet spot exists, use that alpha
    - If cosine ~1.0 at all alphas → router output is near-zero, not learning useful deltas
    - If prediction changes but cosine is high → router found a sensitive direction (good sign)
    - If cosine drops fast but prediction stays same → router pushes hard but uselessly
    """
    )


if __name__ == "__main__":
    base_model, tokenizer, router, device = load_model_and_router()
    print(f"Model device: {device}")
    run_diagnostic(base_model, tokenizer, router, device)