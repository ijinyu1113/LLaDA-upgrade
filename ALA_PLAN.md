# ALA Router Plan for LLaDA (Fast Track)

Goal
Ship a minimal, testable prototype for Sewon Min without pre-training from
scratch. We keep the pre-trained LLaDA weights and add Mechanism A (ALA Router)
only at the input layer.

Core Idea
Replace the static mask embedding (token id 126336) with prompt-conditioned
"seed" vectors produced by a lightweight Router. This changes only how
`input_ids` become `hidden_states` before the first Transformer block.

Architecture Change
Baseline:
`input_ids -> Embedding Layer -> hidden_states`

ALA LLaDA:
1) Identify prompt tokens and mask tokens.
2) Embed prompt tokens normally.
3) For mask tokens, generate unique seeds using a Router that attends to the
   prompt embeddings.
4) Concatenate prompt + seeds and pass into the Transformer blocks using
   `inputs_embeds`.

Implementation Plan
1) Wrapper Module (No internal Transformer edits)
   - Create `ALALLaDA(nn.Module)` that holds:
     - `base_model`: pre-trained LLaDA
     - `router`: MultiheadAttention or another small cross-attention module
   - In `forward`:
     - Compute `inputs_embeds` with `base_model.get_input_embeddings()`
     - Split prompt/mask by `prompt_lengths`
     - For each sample, run Router:
       `seeds = router(query=mask_queries, key=prompt_context, value=prompt_context)`
     - Replace mask embeddings with `seeds`
     - Call `base_model(inputs_embeds=inputs_embeds)`

2) Diagnostic (No training required)
   - Run a single forward pass with the base model vs ALA wrapper.
   - Compute Effective Rank on hidden states:
     - Baseline should be ~1 (near-degenerate for static mask)
     - ALA should be noticeably higher even with random Router weights

3) Training Strategy (Light SFT)
   - Freeze all base LLaDA parameters.
   - Train only the Router on masked tokens.
   - Loss: standard cross-entropy on masked indices.
   - Evaluate on OOD Nim states after a few hours of training.

Deliverables for the Meeting
1) ALA wrapper module that loads existing LLaDA weights.
2) A diagnostic script that compares Effective Rank.
3) Optional: short SFT run (Router-only) + Nim evaluation.

Risks and Mitigations
- Risk: prompt_lengths handling bug -> add assertions + test on variable lengths.
- Risk: Router too small -> increase heads or add a small MLP projection.
- Risk: throughput regression -> batch the Router operation where possible.

How to Start (Quick Instructions)
1) Set up a Python environment and install dependencies.
   - Create venv (example):
     - `python -m venv .venv`
     - `.venv\Scripts\Activate.ps1`
   - Install requirements (repo root):
     - `pip install -r requirements.txt` (if present)
     - For OpenCompass: `pip install -r opencompass/requirements.txt`

2) Download LLaDA weights and confirm base inference works.
   - Follow `README.md` and `EVAL.md` for the exact model path and usage.

3) Implement the wrapper.
   - Add a new file (e.g., `ala_wrapper.py`) with `ALALLaDA`.
   - Load the base model and wrap it before inference.

4) Run the diagnostic.
   - Add a small script (e.g., `ala_diagnostic.py`) to:
     - Run baseline LLaDA
     - Run ALA LLaDA
     - Compute Effective Rank

5) (Optional) Train Router only.
   - Freeze `base_model.parameters()`
   - Optimize `router.parameters()` with SFT loss
   - Evaluate on Nim after a short run

Success Criteria (Minimal)
- ALA wrapper runs with pre-trained weights.
- Effective Rank for masks increases vs baseline.
- No regression in prompt-only behavior.
