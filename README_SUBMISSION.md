# Submission guide

Package layout = repository layout. `submission.json` is the manifest (file list with SHA-256, base model and adapter details,
fine-tuning hyperparameters, honesty disclosures).

## Run a question with the submitted model (Apple silicon Mac)
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt mlx-lm transformers
# base model (4-bit MLX Qwen3-8B) is downloaded once from Hugging Face: mlx-community/Qwen3-8B-4bit
# tokenizer: mlx-community/Qwen3-4B-4bit (same Qwen3 tokenizer and chat template)
python llm/agent.py "<question>" --model mlx:bist8b-v2
```
The LoRA adapter is `llm/adapters8b_v2/adapters.safetensors` (SHA-256 in `submission.json`). Seed 42, temperature 0.2.

## Reproduce the traces
```bash
python llm/build_rag.py
python llm/make_traces.py --model mlx:bist8b-v2      # writes data/events/execution_traces.json
```
Each trace holds `raw_answer` (model output) and `final_answer` (after the post-answer guard, `llm/guard.py`).
