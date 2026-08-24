# hermes-llm-transmission

Per-category sampling parameter tuning for **Hermes 3 3B** (Llama 3.2 3B architecture) on an 8 GB Jetson, comparing **Q4_K_M** and **Q5_K_M** GGUF quantizations via llama.cpp.

## Summary

- 12 diverse prompts (tool calls, prose, email, SQL, poetry, math, HTML, Python) benchmarked at default and tuned settings
- 88-run variable sweep per model (temp, top_p, top_k, repeat_penalty)
- Auto-classifier applies best settings per prompt category
- Context set to 64K (Q4 KV cache to fit Jetson memory)

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
| `transmission_proxy.py` | 173 | Flask proxy: auto-tunes params per prompt, forwards to llama-server |
| `start_transmission.sh` | 93 | One-command start: llama-server + proxy |
| `stop_transmission.sh` | 20 | One-command stop: kills both servers |
| `prompts/01-12_*.txt` | — | 12 benchmark prompt files |
| `results/benchmark_default.json` | — | Default benchmark results (Q4 + Q5) |
| `results/sweep_q4.json` | — | 88-run sweep data for Q4 |
| `results/sweep_q5.json` | — | 88-run sweep data for Q5 |
| `results/benchmark_tuned.json` | — | Tuned retest results (Q4 + Q5) |

## Memory budget (8 GB Jetson, 64K context, Q4 KV cache)

| Component | Q4_K_M | Q5_K_M |
|-----------|--------|--------|
| Weights | 1.90 GB | 2.30 GB |
| KV cache (Q4_0, 64K) | ~3.75 GB | ~3.75 GB |
| Total | ~5.65 GB | ~6.05 GB |
| Free headroom | ~1.7 GB | ~1.3 GB |

Q5_K_M fits with ~1.3 GB headroom when GUI is disabled and Ollama is stopped.

**Q4 KV cache** is the key to running 64K context on an 8 GB Jetson:
- 64K FP16 KV cache would need ~7.5 GB (does not fit)
- 64K Q8 KV cache would need ~3.75 GB (fits but tight, was used for 32K originally)
- 64K Q4 KV cache needs ~3.75 GB (fits comfortably, same as 32K Q8)
- Quality impact of Q4 KV cache is negligible for a 3B model

## Architecture

Hermes 3 3B = Llama 3.2 3B:
- 28 layers, 8 KV heads (GQA), 128 head dim
- 3072 hidden dim, 24 attention heads
- ChatML format (`<|im_start|>...<|im_end|>`)
- 64K context (Q4 KV cache for Jetson memory fit)

## License

MIT

## Hermes Agent Integration (Transmission Proxy)

The **transmission proxy** (`transmission_proxy.py`) connects this tuning
pipeline directly to Hermes Agent — so every prompt Hermes sends is
auto-classified and gets the empirically-best sampling parameters injected
before reaching the model.

### Architecture

```
Hermes Agent
    │  (OpenAI-compatible API)
    ▼
┌──────────────────────┐
│  transmission_proxy   │  Flask :8081
│  (hermes_auto.py      │  Classifies prompt → injects tuned params
│   classifier)         │  → forwards to llama-server
└──────────┬───────────┘
            ▼
┌──────────────────────┐
│  llama-server         │  :8080
│  (llama.cpp)          │  Q5_K_M model, 64K context, Q4 KV cache
│  hermes3-3b-q5_k_m    │  -ngl 99 -fa on --jinja
└──────────────────────┘
```

### Why Q4 KV cache for 64K context

Hermes Agent enforces a **64K minimum context** (`MINIMUM_CONTEXT_LENGTH = 64_000`
in `agent/model_metadata.py`). On an 8 GB Jetson with unified memory, a 64K
FP16 KV cache (~7.5 GB) would not fit alongside the 2.3 GB Q5 model.

Quantizing the KV cache to **Q4_0** halves it to ~3.75 GB — same memory as a
32K Q8 cache. Total footprint: ~2.3 GB model + ~3.75 GB KV = ~6.05 GB, leaving
~1.3 GB headroom with GUI disabled.

Q4 KV cache has minimal quality impact for a 3B model at this context length.

### Prerequisites

- Ollama **stopped** (frees ~1.2 GB GPU memory): `pkill -f ollama`
- llama.cpp built with CUDA: `~/llama.cpp/build/bin/llama-server`
- Q5 model at: `~/models/hermes3-3b-q5_k_m.gguf`
- GUI disabled (recommended): `sudo systemctl set-default multi-user.target`

### Quick start

```bash
# 1. Stop Ollama if running (frees GPU memory)
pkill -f ollama

# 2. Start the full transmission stack (llama-server + proxy)
cd ~/projects/hermes-llm-transmission
./start_transmission.sh

# 3. Point Hermes at the local proxy
hermes config set model.default hermes3-3b-q5
hermes config set model.provider custom
hermes config set model.base_url http://localhost:8081/v1
hermes config set model.api_key local
hermes config set model.context_length 65536

# 4. Test
hermes chat -q "What is 2+2?"
```

### Stop

```bash
./stop_transmission.sh
```

### Revert to cloud model

```bash
# Backup was saved at config.yaml.cloud-backup
cp ~/.hermes/config.yaml.cloud-backup ~/.hermes/config.yaml
# Or manually:
hermes config set model.default glm-5.2
hermes config set model.provider ollama-cloud
hermes config set model.base_url https://ollama.com/v1
hermes config set model.context_length 128000
```

### Implementation steps (how this was built)

1. **Stopped Ollama** — `pkill -f ollama` to free GPU/CPU memory on the Jetson.

2. **Started llama-server** with Q5 model, 64K context, Q4 KV cache:
   ```
   ~/llama.cpp/build/bin/llama-server \
     -m ~/models/hermes3-3b-q5_k_m.gguf \
     --port 8080 -c 65536 -ngl 99 -t 6 \
     -fa on -ctk q4_0 -ctv q4_0 --jinja
   ```
   - Q4 KV cache was the key discovery: 64K Q4 uses the same memory as 32K Q8,
     satisfying Hermes' 64K minimum without exceeding Jetson RAM.

3. **Built `transmission_proxy.py`** — a Flask proxy on :8081 that:
   - Intercepts each `/v1/chat/completions` request from Hermes
   - Extracts the last user message
   - Classifies it via `hermes_auto.py` (tool/prose/email/sqlite/poetry/math/html/python)
   - Overrides `temperature`, `top_p`, `top_k`, `repeat_penalty` with tuned values
   - Forwards to `llama-server` on :8080 (streaming and non-streaming)
   - Also serves `/v1/models` so Hermes sees a valid model ID

4. **Configured Hermes** to use the proxy:
   ```yaml
   model:
     default: hermes3-3b-q5
     provider: custom
     base_url: http://localhost:8081/v1
     api_key: local
     context_length: 65536
   ```

5. **Created startup scripts** — `start_transmission.sh` and `stop_transmission.sh`
   for one-command start/stop of the full stack.

6. **Verified end-to-end** — `hermes chat -q "What is 2+2?"` returns a correct
   response, with the proxy log showing the auto-classifier routing each prompt
   to its tuned parameter set.