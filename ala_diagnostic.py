import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from router_train import ALALLaDA, _calculate_effective_rank

def validate():
    model_id = "GSAI-ML/LLaDA-8B-Instruct"
    ckpt_path = "./ala_results/ala_router_final_500.pt"
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    
    # Load ALA Wrapper
    model = ALALLaDA(base_model).to(device).to(torch.bfloat16)
    model.router.load_state_dict(torch.load(ckpt_path))
    model.eval()

    # Test Prompt
    prompt_text = "Explain the concept of backpropagation in one sentence."
    mask_token_id = 126336
    num_masks = 64

    ids = tokenizer(prompt_text)["input_ids"]
    full_ids = torch.tensor(ids + [mask_token_id] * num_masks).unsqueeze(0).to(device)
    p_len = torch.tensor([len(ids)]).to(device)

    with torch.no_grad():
        # Baseline
        base_out = base_model(input_ids=full_ids, output_hidden_states=True)
        base_r = _calculate_effective_rank(base_out.hidden_states[-1], len(ids))
        
        # ALA
        ala_out = model(input_ids=full_ids, prompt_lengths=p_len, output_hidden_states=True)
        ala_r = _calculate_effective_rank(ala_out.hidden_states[-1], len(ids))

    print(f"\n--- Validation Results ---")
    print(f"Baseline Rank: {base_r:.2f}")
    print(f"Trained ALA Rank: {ala_r:.2f}")
    print(f"Improvement: {((ala_r/base_r)-1)*100:.1f}%")

if __name__ == "__main__":
    validate()