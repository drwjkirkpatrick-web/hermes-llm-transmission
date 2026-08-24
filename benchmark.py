#!/usr/bin/env python3
"""
Benchmark Hermes 3 3B (Q4_K_M and Q5_K_M) on 12 diverse prompts.

Tests both quantizations with DEFAULT llama.cpp settings (before tuning).
Context is ALWAYS 32768 (per user instruction: "keep the 32k context permanent").

Prompts cover:
  1-5:  Tool calls (web search, SQL, calendar, email, file management)
  6:   Prose (clinical explanation)
  7:   Email draft
  8:   SQLite code generation
  9:   Iambic pentameter
  10:  Math proof
  11:  HTML coding
  12:  Python coding

Each run captures: generation speed (tok/s), prompt speed, wall time,
output length, and the raw output for later quality scoring.

Usage:
  python3 benchmark.py                  # test both models
  python3 benchmark.py --model q4       # test only Q4
  python3 benchmark.py --model q5      # test only Q5
  python3 benchmark.py --prompt 1      # test only prompt 1 on both models
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(SCRIPT_DIR, "prompts")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

LLAMA_CLI = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")

# Models — Q4_K_M and Q5_K_M
MODELS = {
    "q4": os.path.expanduser("~/models/hermes3-3b-q4_k_m.gguf"),
    "q5": os.path.expanduser("~/models/hermes3-3b-q5_k_m.gguf"),
}

# Fixed settings (per user: 32K context, GUI off)
CONTEXT = 32768
THREADS = 6
N_TOKENS = 4096       # enough for a complete answer, keeps total runtime manageable

# Default (before-tuning) llama.cpp sampling settings
DEFAULT_TEMP = 0.8
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 40
DEFAULT_REPEAT_PENALTY = 1.1

# ── Prompt definitions ────────────────────────────────────────────────
# (id, category, label, filename)
PROMPT_DEFS = [
    ("01", "tool",       "Tool: Web Search",         "01_tool_web_search.txt"),
    ("02", "tool",       "Tool: SQL Query",          "02_tool_sql_query.txt"),
    ("03", "tool",       "Tool: Calendar",           "03_tool_calendar.txt"),
    ("04", "tool",       "Tool: Send Email",         "04_tool_send_email.txt"),
    ("05", "tool",       "Tool: File Management",    "05_tool_file_management.txt"),
    ("06", "prose",      "Prose (Hashimoto's)",      "06_prose.txt"),
    ("07", "email",      "Email Draft",              "07_email_draft.txt"),
    ("08", "sqlite",     "SQLite Code Gen",          "08_sqlite.txt"),
    ("09", "poetry",     "Iambic Pentameter",        "09_iambic_pentameter.txt"),
    ("10", "math",       "Math Proof (sqrt(2))",     "10_math_proof.txt"),
    ("11", "html",       "HTML/CSS Coding",          "11_html.txt"),
    ("12", "python",     "Python Coding",            "12_python.txt"),
]

# ── Hermes 3 ChatML system prompt ─────────────────────────────────────
# Hermes 3 uses ChatML format. We provide a neutral system prompt.
SYSTEM_PROMPT = "You are Hermes 3, a helpful and knowledgeable AI assistant. You follow instructions carefully and provide accurate, well-structured responses."

def load_prompt(filename: str) -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path) as f:
        return f.read().strip()

def build_chatml_prompt(system: str, user: str) -> str:
    """Build a ChatML-formatted prompt for Hermes 3."""
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

def run_llama_cli(model_path: str, prompt: str, temp: float = DEFAULT_TEMP,
                  top_p: float = DEFAULT_TOP_P, top_k: int = DEFAULT_TOP_K,
                  repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
                  n_tokens: int = N_TOKENS, context: int = CONTEXT) -> dict:
    """Run llama-cli with given settings, return parsed output."""
    full_prompt = build_chatml_prompt(SYSTEM_PROMPT, prompt)

    cmd = [
        LLAMA_CLI,
        "-m", model_path,
        "-p", full_prompt,
        "-n", str(n_tokens),
        "-c", str(context),
        "--temp", str(temp),
        "--top-p", str(top_p),
        "--top-k", str(top_k),
        "--repeat-penalty", str(repeat_penalty),
        "-t", str(THREADS),
        "-ngl", "99",
        "-fa", "on",
        "--jinja",
        "--no-conversation",
        "--no-display-prompt",
        "-st",
    ]

    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        wall_time = time.time() - start
        combined = result.stdout + result.stderr
        return parse_output(combined, wall_time)
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "wall_time_s": time.time() - start}
    except Exception as e:
        return {"error": str(e), "wall_time_s": time.time() - start}

def parse_output(text: str, wall_time: float) -> dict:
    """Extract metrics from llama-cli stdout+stderr."""
    gen_tps = 0.0
    prompt_tps = 0.0
    total_tokens = 0

    m = re.search(r"Generation:\s*([\d.]+)\s*t/s", text)
    if m:
        gen_tps = float(m.group(1))
    m = re.search(r"Prompt:\s*([\d.]+)\s*t/s", text)
    if m:
        prompt_tps = float(m.group(1))
    m = re.search(r"n_eval\s*=\s*(\d+)", text)
    if m:
        total_tokens = int(m.group(1))

    # Extract the generated text — find the actual assistant response
    output = text
    # The model output appears after the last "> assistant\n" echo from llama-cli
    # Find the LAST occurrence of "> assistant" followed by content
    # The actual generation is between the last "assistant\n" and the timing stats
    parts = output.split('> assistant')
    if len(parts) > 1:
        # Take everything after the LAST "> assistant" marker
        output = parts[-1]
    else:
        # Fallback: try finding "assistant\n" without the "> " prefix
        m = re.search(r'(?:^|\n)assistant\s*\n+', output)
        if m:
            output = output[m.end():]

    # Remove llama.cpp banner/chrome that may appear before the actual text
    # Strip leading spinner artifacts, ASCII art, etc.
    output = re.sub(r'^[\s\\|/-]*\n', '', output)
    output = re.sub(r'^[▄█▀\s]+\n', '', output)
    # Remove "model :", "ftype :", "modalities :", "available commands:" blocks
    output = re.sub(r'^model\s*:.*$', '', output, flags=re.MULTILINE)
    output = re.sub(r'^ftype\s*:.*$', '', output, flags=re.MULTILINE)
    output = re.sub(r'^modalities\s*:.*$', '', output, flags=re.MULTILINE)
    output = re.sub(r'^available commands:.*$', '', output, flags=re.MULTILINE)
    output = re.sub(r'^\s*/\w+\s+.*$', '', output, flags=re.MULTILINE)
    output = re.sub(r'^build\s*:.*$', '', output, flags=re.MULTILINE)
    output = re.sub(r'^system_info\s*:.*$', '', output, flags=re.MULTILINE)
    output = re.sub(r'^main\s*:.*$', '', output, flags=re.MULTILINE)
    # Remove timing lines
    output = re.sub(r'^\s*(Prompt|Generation|Total|Per Token|Eval|Print|n_eval|n_token):.*$', '', output, flags=re.MULTILINE)
    # Remove sampler info
    output = re.sub(r'^\s*sampler.*$', '', output, flags=re.MULTILINE)
    # Remove the echoed prompt content (system/user lines from chatml echo)
    output = re.sub(r'^>\s*system\s*$', '', output, flags=re.MULTILINE)
    output = re.sub(r'^>\s*user\s*$', '', output, flags=re.MULTILINE)
    # Remove ChatML tokens
    for marker in ('<|im_start|>', '<|im_end|>'):
        output = output.replace(marker, '')
    # Clean up excessive blank lines from removed content
    output = re.sub(r'\n{3,}', '\n\n', output)

    output = output.strip()

    output_chars = len(output)

    return {
        "gen_tps": gen_tps,
        "prompt_tps": prompt_tps,
        "total_tokens": total_tokens,
        "output_chars": output_chars,
        "wall_time_s": round(wall_time, 1),
        "output": output[:5000],  # cap stored output for JSON size
    }

def benchmark_model(model_key: str, prompt_filter: str = None) -> list:
    """Run all (or filtered) prompts on one model."""
    model_path = MODELS[model_key]
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return []

    results = []
    for pid, category, label, filename in PROMPT_DEFS:
        if prompt_filter and pid != prompt_filter:
            continue

        prompt_text = load_prompt(filename)
        print(f"\n  [{model_key}] {pid}: {label}...", end="", flush=True)

        r = run_llama_cli(model_path, prompt_text)
        r["id"] = pid
        r["model"] = model_key
        r["category"] = category
        r["label"] = label
        r["settings"] = {
            "temp": DEFAULT_TEMP,
            "top_p": DEFAULT_TOP_P,
            "top_k": DEFAULT_TOP_K,
            "repeat_penalty": DEFAULT_REPEAT_PENALTY,
            "context": CONTEXT,
            "n_tokens": N_TOKENS,
        }

        if "error" in r:
            print(f" ERROR: {r['error']}")
        else:
            print(f" {r['gen_tps']:.1f} t/s, {r['output_chars']} chars, {r['wall_time_s']}s")

        results.append(r)

    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark Hermes 3 3B Q4 vs Q5")
    parser.add_argument("--model", choices=["q4", "q5", "both"], default="both")
    parser.add_argument("--prompt", type=str, help="Only test this prompt ID (e.g. '01')")
    args = parser.parse_args()

    all_results = {
        "meta": {
            "models": {k: os.path.basename(v) for k, v in MODELS.items()},
            "settings": {
                "temp": DEFAULT_TEMP,
                "top_p": DEFAULT_TOP_P,
                "top_k": DEFAULT_TOP_K,
                "repeat_penalty": DEFAULT_REPEAT_PENALTY,
                "context": CONTEXT,
                "n_tokens": N_TOKENS,
                "threads": THREADS,
            },
            "prompts": len(PROMPT_DEFS),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

    models_to_test = ["q4", "q5"] if args.model == "both" else [args.model]

    # Load existing results if present (so --model q4 and --model q5 can be run separately)
    outfile = os.path.join(RESULTS_DIR, "benchmark_default.json")
    if os.path.exists(outfile):
        try:
            with open(outfile) as f:
                old = json.load(f)
            # Preserve meta and any existing model results
            if "meta" in old:
                all_results["meta"] = old["meta"]
            for mk in ["q4", "q5"]:
                if mk in old and mk not in models_to_test:
                    all_results[mk] = old[mk]
        except Exception:
            pass  # Corrupt or old format — start fresh

    for mk in models_to_test:
        print(f"\n{'='*70}")
        print(f"  Benchmarking {mk.upper()} ({os.path.basename(MODELS[mk])})")
        print(f"  Settings: temp={DEFAULT_TEMP} top_p={DEFAULT_TOP_P} top_k={DEFAULT_TOP_K} rep={DEFAULT_REPEAT_PENALTY}")
        print(f"  Context: {CONTEXT} | Tokens: {N_TOKENS} | Threads: {THREADS}")
        print(f"{'='*70}")
        all_results[mk] = benchmark_model(mk, args.prompt)

    # Save results
    outfile = os.path.join(RESULTS_DIR, "benchmark_default.json")
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outfile}")

    # Print summary table
    print(f"\n{'='*90}")
    print(f"{'ID':<5} {'Cat':<10} {'Label':<28} {'Q4 t/s':>8} {'Q5 t/s':>8} {'Q4 ch':>7} {'Q5 ch':>7} {'Q5-Q4':>6}")
    print(f"{'-'*90}")
    for i, (pid, cat, label, _) in enumerate(PROMPT_DEFS):
        if args.prompt and pid != args.prompt:
            continue
        q4 = all_results.get("q4", [])
        q5 = all_results.get("q5", [])
        q4r = next((r for r in q4 if r["id"] == pid), None)
        q5r = next((r for r in q5 if r["id"] == pid), None)
        q4tps = f"{q4r['gen_tps']:.1f}" if q4r and "gen_tps" in q4r else "—"
        q5tps = f"{q5r['gen_tps']:.1f}" if q5r and "gen_tps" in q5r else "—"
        q4ch = q4r["output_chars"] if q4r and "output_chars" in q4r else "—"
        q5ch = q5r["output_chars"] if q5r and "output_chars" in q5r else "—"
        diff = (q5r["output_chars"] - q4r["output_chars"]) if (q4r and q5r and "output_chars" in q4r and "output_chars" in q5r) else 0
        print(f"{pid:<5} {cat:<10} {label:<28} {q4tps:>8} {q5tps:>8} {str(q4ch):>7} {str(q5ch):>7} {diff:>+6}")

    print(f"{'='*90}")

if __name__ == "__main__":
    main()