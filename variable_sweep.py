#!/usr/bin/env python3
"""
Variable sweep for Hermes 3 3B — organized fine-tuning tests.

Tests ONE variable at a time, holding everything else fixed at known-best
values. Context is ALWAYS 32768 (per user instruction). Temperature uses
the per-category best from the temperature sweep.

Variables tested:
  1. temperature:     0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0   (7 values)
  2. top_p:            0.80, 0.90, 0.95, 0.98, 1.0          (5 values)
  3. top_k:             0,   20,   40,   60,   80           (5 values)
  4. repeat_penalty:    1.0,  1.05, 1.1,  1.15, 1.2        (5 values)

Prompts (4 representative categories):
  tool:      "Write the JSON tool call to search the web..."  (temp 0.7)
  prose:     "Explain the pathophysiology of Hashimoto's..."   (temp 0.5)
  poetry:    "Write a sonnet in iambic pentameter..."           (temp 0.3)
  math:      "Prove that sqrt(2) is irrational..."              (temp 1.0)

Total: 4 prompts x (7 + 5 + 5 + 5) = 4 x 22 = 88 runs per model.
Token budget: 2048 (enough for a complete answer, keeps total runtime manageable).
Context: 32768 (fixed per user instruction).

Usage:
  python3 variable_sweep.py                       # sweep both models
  python3 variable_sweep.py --model q4            # sweep only Q4
  python3 variable_sweep.py --model q5           # sweep only Q5
  python3 variable_sweep.py --phase temp          # only temperature sweep
  python3 variable_sweep.py --phase top_p          # only top_p sweep
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

LLAMA_CLI = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")
MODELS = {
    "q4": os.path.expanduser("~/models/hermes3-3b-q4_k_m.gguf"),
    "q5": os.path.expanduser("~/models/hermes3-3b-q5_k_m.gguf"),
}

CONTEXT = 32768
THREADS = 6
N_TOKENS = 2048      # shorter for sweep — enough for quality assessment

SYSTEM_PROMPT = "You are Hermes 3, a helpful and knowledgeable AI assistant. You follow instructions carefully and provide accurate, well-structured responses."

# ── Representative prompts (one per category) ─────────────────────────
PROMPTS = [
    {
        "id": "tool",
        "label": "Tool Call (web search JSON)",
        "text": 'You are a research assistant with access to a web_search tool. A user asks: "What are the latest findings on the gut microbiome\'s role in autoimmune thyroid conditions?" Write the exact tool call you would make as a JSON object with a "tool_calls" array. Include the search query, parameters, and how you would process the results.',
        "temp": 0.7,
    },
    {
        "id": "prose",
        "label": "Prose (Hashimoto's pathophysiology)",
        "text": "Explain the pathophysiology of Hashimoto's thyroiditis in detail. Cover the autoimmune mechanism, the role of anti-TPO antibodies, the progression from euthyroid to hypothyroid, and the typical lab findings at each stage. Write for a medical student audience.",
        "temp": 0.5,
    },
    {
        "id": "poetry",
        "label": "Iambic Pentameter Sonnet",
        "text": "Write a poem about the healing power of nature and the changing of seasons, strictly in iambic pentameter (10 syllables per line, alternating stress: da-DUM da-DUM da-DUM da-DUM da-DUM). Write exactly 14 lines as a Shakespearean sonnet with an ABAB CDCD EFEF GG rhyme scheme. Do not include any commentary, just the 14 lines of the poem.",
        "temp": 0.3,
    },
    {
        "id": "math",
        "label": "Math Proof (sqrt(2) irrational)",
        "text": 'Prove that the square root of 2 is irrational. Use a proof by contradiction. State each step clearly with justification. Begin with: "Theorem: sqrt(2) is irrational."',
        "temp": 1.0,
    },
]

# ── Variable definitions ──────────────────────────────────────────────
DEFAULTS = {
    "temp": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "repeat_penalty": 1.1,
}

SWEEPS = [
    {
        "variable": "temp",
        "label": "Temperature",
        "values": [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0],
        "description": "Controls randomness. Lower = more deterministic/focused. Higher = more creative/diverse.",
    },
    {
        "variable": "top_p",
        "label": "Nucleus sampling (top_p)",
        "values": [0.80, 0.90, 0.95, 0.98, 1.0],
        "description": "Controls how many of the most-likely tokens are considered. Lower = more focused. Higher = more diverse.",
    },
    {
        "variable": "top_k",
        "label": "Top-K sampling (top_k)",
        "values": [0, 20, 40, 60, 80],
        "description": "Limits to the K most-likely tokens. 0 = disabled. Lower K = more focused. Higher K = more diverse.",
    },
    {
        "variable": "repeat_penalty",
        "label": "Repetition penalty (repeat_penalty)",
        "values": [1.0, 1.05, 1.1, 1.15, 1.2],
        "description": "Penalizes tokens that already appeared. 1.0 = no penalty. Higher = more penalty, discourages repetition.",
    },
]

# ── Core runner (imported from benchmark) ──────────────────────────────
def build_chatml_prompt(system, user):
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

def run_one(model_path, prompt_text, temp, top_p, top_k, repeat_penalty,
            n_tokens=N_TOKENS, context=CONTEXT):
    full_prompt = build_chatml_prompt(SYSTEM_PROMPT, prompt_text)
    cmd = [
        LLAMA_CLI, "-m", model_path, "-p", full_prompt,
        "-n", str(n_tokens), "-c", str(context),
        "--temp", str(temp), "--top-p", str(top_p),
        "--top-k", str(top_k), "--repeat-penalty", str(repeat_penalty),
        "-t", str(THREADS), "-ngl", "99", "-fa", "on",
        "--jinja", "--no-conversation", "--no-display-prompt", "-st",
    ]
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        wall_time = time.time() - start
        combined = result.stdout + result.stderr
        return parse_output(combined, wall_time)
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "wall_time_s": round(time.time() - start, 1)}
    except Exception as e:
        return {"error": str(e), "wall_time_s": round(time.time() - start, 1)}

def parse_output(text, wall_time):
    gen_tps = 0.0
    m = re.search(r"Generation:\s*([\d.]+)\s*t/s", text)
    if m:
        gen_tps = float(m.group(1))

    output = text
    # Use the improved parser: split on "> assistant" and take the last part
    parts = output.split('> assistant')
    if len(parts) > 1:
        output = parts[-1]
    else:
        m = re.search(r'(?:^|\n)assistant\s*\n+', output)
        if m:
            output = output[m.end():]

    output = re.sub(r'^[\s\\|/-]*\n', '', output)
    output = re.sub(r'^[▄█▀\s]+\n', '', output)
    output = re.sub(r'^model\s*:.*$', '', output, flags=re.MULTILINE)
    output = re.sub(r'^ftype\s*:.*$', '', output, flags=re.MULTILINE)
    output = re.sub(r'^modalities\s*:.*$', '', output, flags=re.MULTILINE)
    output = re.sub(r'^available commands:.*$', '', output, flags=re.MULTILINE)
    output = re.sub(r'^\s*/\w+\s+.*$', '', output, flags=re.MULTILINE)
    output = re.sub(r'^\s*(build|system_info|main|sampler|Prompt|Generation|Total|Per Token|Eval|Print|n_eval|n_token):.*$', '', output, flags=re.MULTILINE)
    output = re.sub(r'^>\s*system\s*$', '', output, flags=re.MULTILINE)
    output = re.sub(r'^>\s*user\s*$', '', output, flags=re.MULTILINE)
    for marker in ('<|im_start|>', '<|im_end|>'):
        output = output.replace(marker, '')
    output = re.sub(r'\n{3,}', '\n\n', output)
    output = output.strip()

    return {
        "gen_tps": gen_tps,
        "output_chars": len(output),
        "wall_time_s": round(wall_time, 1),
        "output": output[:3000],
    }

# ── Quality scoring (heuristic, automated) ────────────────────────────
def score_quality(prompt_id, output):
    """Heuristic quality score 0-10 based on output characteristics."""
    if not output or len(output) < 10:
        return 0

    score = 5.0  # base
    length = len(output)

    # Length appropriateness (not too short, not truncated)
    if length > 100:
        score += 1.0
    if length > 500:
        score += 0.5
    if length > 2000:
        score -= 0.5  # too verbose
    if length < 50:
        score -= 2.0

    # Category-specific checks
    if prompt_id == "poetry":
        lines = [l for l in output.split('\n') if l.strip()]
        if 12 <= len(lines) <= 16:
            score += 2.0
        elif 8 <= len(lines) <= 20:
            score += 1.0
        else:
            score -= 1.0
        # Check syllable count (rough: 10 syllables per line)
        syllable_ok = 0
        for line in lines[:14]:
            words = line.split()
            if 8 <= len(words) <= 12:
                syllable_ok += 1
        if lines:
            ratio = syllable_ok / min(len(lines), 14)
            score += ratio * 2.0

    elif prompt_id == "math":
        # Check for proof structure keywords
        has_contradiction = bool(re.search(r'contradict', output, re.I))
        has_assume = bool(re.search(r'assum', output, re.I))
        has_even_odd = bool(re.search(r'even|odd|2[dk]', output, re.I))
        has_qed = bool(re.search(r'Q\.?E\.?D|\u25a0|therefore|thus', output, re.I))
        for check in [has_contradiction, has_assume, has_even_odd, has_qed]:
            if check:
                score += 1.0

    elif prompt_id == "tool":
        # Check for JSON structure
        has_json = bool(re.search(r'[\{\[]', output))
        has_tool_name = bool(re.search(r'(web_search|search|query)', output, re.I))
        has_args = bool(re.search(r'(argument|param|query.*:)', output, re.I))
        for check in [has_json, has_tool_name, has_args]:
            if check:
                score += 1.0

    elif prompt_id == "prose":
        # Check for medical content keywords
        keywords = re.findall(r'(thyroid|antibod|TPO|hypothyroid|euthyroid|autoimmune|TSH|lymphocyte)', output, re.I)
        if len(keywords) >= 5:
            score += 2.0
        elif len(keywords) >= 3:
            score += 1.0

    # Repetition penalty
    words = output.lower().split()
    if len(words) > 50:
        unique = len(set(words))
        ratio = unique / len(words)
        if ratio < 0.3:
            score -= 2.0  # heavy repetition
        elif ratio < 0.5:
            score -= 1.0

    return max(0, min(10, round(score, 1)))

# ── Main sweep logic ──────────────────────────────────────────────────
def run_sweep(model_key, phase_filter=None):
    model_path = MODELS[model_key]
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return []

    all_runs = []

    for sweep in SWEEPS:
        if phase_filter and sweep["variable"] != phase_filter:
            continue

        var = sweep["variable"]
        print(f"\n{'='*70}")
        print(f"  Sweeping {sweep['label']} on {model_key.upper()}")
        print(f"  Values: {sweep['values']}")
        print(f"  {sweep['description']}")
        print(f"{'='*70}")

        for prompt in PROMPTS:
            for val in sweep["values"]:
                # Build settings: sweep variable changes, others stay at defaults
                settings = dict(DEFAULTS)
                settings[var] = val

                label = f"[{model_key}] {var}={val} | {prompt['id']}"
                print(f"  {label}...", end="", flush=True)

                r = run_one(
                    model_path, prompt["text"],
                    temp=settings["temp"],
                    top_p=settings["top_p"],
                    top_k=settings["top_k"],
                    repeat_penalty=settings["repeat_penalty"],
                )

                r["model"] = model_key
                r["sweep_var"] = var
                r["sweep_val"] = val
                r["prompt_id"] = prompt["id"]
                r["prompt_label"] = prompt["label"]
                r["settings"] = settings
                r["quality_score"] = score_quality(prompt["id"], r.get("output", ""))

                if "error" in r:
                    print(f" ERROR: {r['error']}")
                else:
                    print(f" q={r['quality_score']:.1f} {r['gen_tps']:.1f}t/s {r['output_chars']}ch")

                all_runs.append(r)

    return all_runs

def main():
    parser = argparse.ArgumentParser(description="Variable sweep for Hermes 3 3B")
    parser.add_argument("--model", choices=["q4", "q5", "both"], default="both")
    parser.add_argument("--phase", choices=["temp", "top_p", "top_k", "repeat_penalty"],
                        help="Only run this sweep phase")
    args = parser.parse_args()

    models_to_test = ["q4", "q5"] if args.model == "both" else [args.model]
    all_results = {}

    for mk in models_to_test:
        print(f"\n{'#'*70}")
        print(f"#  Variable Sweep: {mk.upper()} ({os.path.basename(MODELS[mk])})")
        print(f"#  Context: {CONTEXT} | Tokens: {N_TOKENS} | Threads: {THREADS}")
        print(f"{'#'*70}")
        runs = run_sweep(mk, args.phase)
        all_results[mk] = runs

        # Save incrementally
        outfile = os.path.join(RESULTS_DIR, f"sweep_{mk}.json")
        with open(outfile, "w") as f:
            json.dump({
                "meta": {"model": mk, "context": CONTEXT, "n_tokens": N_TOKENS},
                "defaults": DEFAULTS,
                "sweeps": [s["variable"] for s in SWEEPS],
                "runs": runs,
            }, f, indent=2)
        print(f"\nSaved {len(runs)} runs to {outfile}")

    # Print summary
    print(f"\n{'='*90}")
    print("SWEEP SUMMARY — Best quality per variable per prompt")
    print(f"{'='*90}")
    for mk in models_to_test:
        runs = all_results.get(mk, [])
        if not runs:
            continue
        print(f"\n  {mk.upper()}:")
        for var in [s["variable"] for s in SWEEPS]:
            if args.phase and var != args.phase:
                continue
            var_runs = [r for r in runs if r["sweep_var"] == var]
            if not var_runs:
                continue
            # Group by prompt, find best per prompt
            for pid in [p["id"] for p in PROMPTS]:
                pid_runs = [r for r in var_runs if r["prompt_id"] == pid]
                if not pid_runs:
                    continue
                best = max(pid_runs, key=lambda r: r.get("quality_score", 0))
                print(f"    {var:<18} {pid:<10} best={best['sweep_val']:<6} q={best['quality_score']:.1f}  {best['output_chars']}ch {best['gen_tps']:.1f}t/s")

if __name__ == "__main__":
    main()