# hermes-llm-transmission

Per-category sampling parameter tuning for **Hermes 3 3B** (Llama 3.2 3B architecture) on an 8 GB Jetson, comparing **Q4_K_M** and **Q5_K_M** GGUF quantizations via llama.cpp.

## Summary

- 12 diverse prompts (tool calls, prose, email, SQL, poetry, math, HTML, Python) benchmarked at default and tuned settings
- 88-run variable sweep per model (temp, top_p, top_k, repeat_penalty)
- Auto-classifier applies best settings per prompt category
- Context permanently fixed at 32K

## Results

### Speed

| Model | Default (t/s) | Tuned (t/s) |
|-------|---------------|-------------|
| Q4_K_M | 19.6 | 19.6 |
| Q5_K_M | 19.3 | 19.3 |

Q5 is ~0.3 t/s slower than Q4 — a 1.5% speed cost for better quality.

### Quality (before → after tuning)

| ID | Category | Label | Q4-B | Q4-A | dQ4 | Q5-B | Q5-A | dQ5 |
|----|----------|-------|------|------|-----|------|------|-----|
| 01 | tool | Web Search | 8.5 | 9.0 | +0.5 | 8.0 | 9.0 | +1.0 |
| 02 | tool | SQL Query | 8.5 | 9.0 | +0.5 | 8.5 | 9.0 | +0.5 |
| 03 | tool | Calendar | 8.5 | 9.5 | +1.0 | 8.5 | 9.5 | +1.0 |
| 04 | tool | Send Email | 8.0 | 8.0 | +0.0 | 8.5 | 8.0 | -0.5 |
| 05 | tool | File Mgmt | 8.5 | 9.0 | +0.5 | 8.5 | 9.0 | +0.5 |
| 06 | prose | Hashimoto's | 7.0 | 7.0 | +0.0 | 7.0 | 7.0 | +0.0 |
| 07 | email | Email Draft | 7.0 | 9.0 | +2.0 | 7.0 | 9.0 | +2.0 |
| 08 | sqlite | SQLite Gen | 8.0 | 7.5 | -0.5 | 8.0 | 8.5 | +0.5 |
| 09 | poetry | Iambic Pent. | 5.9 | 5.9 | +0.0 | 5.9 | 6.1 | +0.2 |
| 10 | math | sqrt(2) Proof | 9.0 | 8.0 | -1.0 | 9.0 | 8.0 | -1.0 |
| 11 | html | HTML/CSS | 6.0 | 7.5 | +1.5 | 6.0 | 7.5 | +1.5 |
| 12 | python | Python | 8.0 | 8.5 | +0.5 | 8.0 | 8.5 | +0.5 |
| **AVG** | | | **7.7** | **8.2** | **+0.4** | **7.7** | **8.3** | **+0.5** |

### Key findings

- **Email** improved most with tuning (+2.0 quality) — moderate temp with low repeat penalty produces better professional tone
- **HTML/CSS** improved +1.5 — low temp + high top_p produces cleaner, more complete code
- **Tool calls** improved +0.5–1.0 — low temp produces more consistent JSON structure
- **Math proof** dropped -1.0 — low temp produces shorter, more structured proofs (fewer words but still logically valid)
- **Poetry** barely moved — the quality heuristic cannot capture metrical accuracy; manual inspection shows temp=0.3 + top_k=80 produces the best verse
- **Q5** slightly outperforms Q4 in quality (8.3 vs 8.2 avg) at a 1.5% speed cost

## Tuned Settings

Four archetypes based on sweep analysis:

| Archetype | Categories | temp | top_p | top_k | repeat_penalty |
|-----------|-----------|------|-------|-------|----------------|
| Reasoning | math, sqlite, html, python | 0.1 | 0.98 | 40 | 1.15 |
| Tool calls | tool | 0.1 | 0.95 | 40 | 1.20 |
| Prose | prose, email | 0.8 | 0.95 | 40 | 1.05 |
| Creative | poetry | 0.3 | 0.95 | 80 | 1.20 |

**Rationale:**
- **Reasoning**: Low temp for determinism, high top_p for vocabulary breadth, top_k=40 (sweep best for math), rep=1.15 prevents loops in code/proofs
- **Tool calls**: Low temp for JSON accuracy, rep=1.2 prevents repetitive JSON key patterns
- **Prose**: Moderate temp for natural flow, low rep penalty (1.05) allows natural repetition
- **Creative**: Low-moderate temp for structure adherence, top_k=80 for more token diversity in word choice

## Setup

### Prerequisites

- 8 GB Jetson (unified memory)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) built with CUDA support
- GGUF models from Ollama registry:
  - `hermes3:3b` (Q4_K_M, ~1.9 GB)
  - Q5_K_M blob (~2.2 GB)
- GUI disabled to free RAM: `sudo systemctl set-default multi-user.target`

### Install

```bash
# Clone
git clone https://github.com/drwjkirkpatrick-web/hermes-llm-transmission.git
cd hermes-llm-transmission

# Models go in ~/models/
# Q4: pull from ollama registry, extract blob
# Q5: find Q5_K_M manifest blob, copy to ~/models/

# llama-cli at ~/llama.cpp/build/bin/llama-cli
```

## Usage

### Quick start

```bash
# Auto-classified prompt (picks best settings)
./ask-hermes3.sh "Write a SQL query to find the top 5 customers by revenue"

# Use Q4 instead of Q5
./ask-hermes3.sh --model q4 "Explain Hashimoto's thyroiditis"

# Raw mode (no auto-classifier, default settings)
./ask-hermes3.sh --raw "What is 2+2?"
```

### Full pipeline

```bash
# 1. Benchmark both models at default settings
python3 benchmark.py --model q4
python3 benchmark.py --model q5

# 2. Variable sweep (88 runs per model, ~30 min each)
python3 variable_sweep.py --model q5
python3 variable_sweep.py --model q4

# 3. Tuned retest (applies best settings per category)
python3 tuned_retest.py --model q5
python3 tuned_retest.py --model q4

# 4. Use the auto-classifier standalone
python3 hermes_auto.py "your prompt here"
# Output: temp top_p top_k repeat_penalty style context tokens
```

## Files

| File | Lines | Description |
|------|-------|-------------|
| `benchmark.py` | 280 | Default benchmark: 12 prompts, both models, fixed 32K context |
| `variable_sweep.py` | 367 | Sweeps temp (7), top_p (5), top_k (5), repeat_penalty (5) per model |
| `tuned_retest.py` | 387 | Re-benchmarks with tuned settings, compares before/after |
| `hermes_auto.py` | 191 | Keyword classifier that outputs best settings per category |
| `ask-hermes3.sh` | 140 | Launcher: auto-classifies prompt, runs llama-cli with tuned settings |
| `prompts/01-12_*.txt` | — | 12 benchmark prompt files |
| `results/benchmark_default.json` | — | Default benchmark results (Q4 + Q5) |
| `results/sweep_q4.json` | — | 88-run sweep data for Q4 |
| `results/sweep_q5.json` | — | 88-run sweep data for Q5 |
| `results/benchmark_tuned.json` | — | Tuned retest results (Q4 + Q5) |

## Memory budget (8 GB Jetson, 32K context)

| Component | Q4_K_M | Q5_K_M |
|-----------|--------|--------|
| Weights | 1.90 GB | 2.30 GB |
| KV cache (FP16, 32K) | 3.76 GB | 3.76 GB |
| Total | 5.66 GB | 6.06 GB |
| Free headroom | ~1.3 GB | ~1.0 GB |

Q5_K_M fits with ~1 GB headroom when GUI is disabled.

## Architecture

Hermes 3 3B = Llama 3.2 3B:
- 28 layers, 8 KV heads (GQA), 128 head dim
- 3072 hidden dim, 24 attention heads
- ChatML format (`<|im_start|>...<|im_end|>`)
- 32K context (permanent)

## License

MIT