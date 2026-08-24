#!/usr/bin/env python3
"""
Re-benchmark all 12 prompts with TUNED settings (after sweep).

For each prompt:
  1. Classify it using hermes_auto.py's classifier
  2. Apply the empirically-best settings from the variable sweep
  3. Keep 32K context fixed (per user instruction)
  4. Run both Q4_K_M and Q5_K_M

Then compare default (before) vs tuned (after) quality and speed.

Usage:
  python3 tuned_retest.py                      # both models
  python3 tuned_retest.py --model q4           # only Q4
  python3 tuned_retest.py --model q5          # only Q5
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
MODELS = {
    "q4": os.path.expanduser("~/models/hermes3-3b-q4_k_m.gguf"),
    "q5": os.path.expanduser("~/models/hermes3-3b-q5_k_m.gguf"),
}

CONTEXT = 32768
THREADS = 6
N_TOKENS = 4096

SYSTEM_PROMPT = "You are Hermes 3, a helpful and knowledgeable AI assistant. You follow instructions carefully and provide accurate, well-structured responses."

# ── Tuned settings per category ────────────────────────────────────────
# Empirically derived from Q5_K_M variable sweep (88 runs) on Jetson.
# See hermes_auto.py for detailed rationale per archetype.
TUNED = {
    "tool":      {"temp": 0.1,  "top_p": 0.95, "top_k": 40,  "repeat_penalty": 1.20},
    "prose":     {"temp": 0.8,  "top_p": 0.95, "top_k": 40,  "repeat_penalty": 1.05},
    "email":     {"temp": 0.8,  "top_p": 0.95, "top_k": 40,  "repeat_penalty": 1.05},
    "sqlite":    {"temp": 0.1,  "top_p": 0.98, "top_k": 40,  "repeat_penalty": 1.15},
    "poetry":    {"temp": 0.3,  "top_p": 0.95, "top_k": 80,  "repeat_penalty": 1.20},
    "math":      {"temp": 0.1,  "top_p": 0.98, "top_k": 40,  "repeat_penalty": 1.15},
    "html":      {"temp": 0.1,  "top_p": 0.98, "top_k": 40,  "repeat_penalty": 1.15},
    "python":    {"temp": 0.1,  "top_p": 0.98, "top_k": 40,  "repeat_penalty": 1.15},
}

# ── Prompt definitions (same as benchmark.py) ──────────────────────────
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

# ── Load tuned settings from sweep results if available ───────────────
def load_tuned_from_sweep():
    """Try to load the best settings from sweep results."""
    for mk in ["q5", "q4"]:
        path = os.path.join(RESULTS_DIR, f"sweep_{mk}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        runs = data.get("runs", [])

        # Group by prompt_id and sweep_var, find best quality per var
        best_per = {}  # (prompt_id, var) -> (val, score)
        for r in runs:
            if "error" in r:
                continue
            pid = r["prompt_id"]
            var = r["sweep_var"]
            val = r["sweep_val"]
            q = r.get("quality_score", 0)
            key = (pid, var)
            if key not in best_per or q > best_per[key][1]:
                best_per[key] = (val, q)

        # Map prompt ids to categories
        pid_to_cat = {p[0]: p[1] for p in PROMPT_DEFS}

        # For each category, find best value for each variable
        cat_best = {}  # category -> {var: (val, score)}
        for (pid, var), (val, score) in best_per.items():
            cat = pid_to_cat.get(pid, pid)
            if cat not in cat_best:
                cat_best[cat] = {}
            cat_best[cat][var] = (val, score)

        # Average across prompts within each category
        # Group by category
        cat_runs = {}  # category -> {var: [(val, score)]}
        for r in runs:
            if "error" in r:
                continue
            pid = r["prompt_id"]
            cat = pid_to_cat.get(pid, pid)
            var = r["sweep_var"]
            val = r["sweep_val"]
            q = r.get("quality_score", 0)
            if cat not in cat_runs:
                cat_runs[cat] = {}
            if var not in cat_runs[cat]:
                cat_runs[cat][var] = {}
            if val not in cat_runs[cat][var]:
                cat_runs[cat][var][val] = []
            cat_runs[cat][var][val].append(q)

        # For each category and variable, find the value with best avg quality
        tuned = {}
        for cat, vars_data in cat_runs.items():
            tuned[cat] = {}
            for var, val_scores in vars_data.items():
                best_val = None
                best_avg = -1
                for val, scores in val_scores.items():
                    avg = sum(scores) / len(scores)
                    if avg > best_avg:
                        best_avg = avg
                        best_val = val
                tuned[cat][var] = best_val

        return tuned

    return None

def build_chatml_prompt(system, user):
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

def run_llama_cli(model_path, prompt_text, settings):
    full_prompt = build_chatml_prompt(SYSTEM_PROMPT, prompt_text)
    cmd = [
        LLAMA_CLI, "-m", model_path, "-p", full_prompt,
        "-n", str(N_TOKENS), "-c", str(CONTEXT),
        "--temp", str(settings["temp"]),
        "--top-p", str(settings["top_p"]),
        "--top-k", str(settings["top_k"]),
        "--repeat-penalty", str(settings["repeat_penalty"]),
        "-t", str(THREADS), "-ngl", "99", "-fa", "on",
        "--jinja", "--no-conversation", "--no-display-prompt", "-st",
    ]
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
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
        "output": output[:5000],
    }

def score_quality(category, output):
    """Same heuristic scoring as variable_sweep.py."""
    if not output or len(output) < 10:
        return 0
    score = 5.0
    length = len(output)
    if length > 100:
        score += 1.0
    if length > 500:
        score += 0.5
    if length > 2000:
        score -= 0.5
    if length < 50:
        score -= 2.0

    if category == "poetry":
        lines = [l for l in output.split('\n') if l.strip()]
        if 12 <= len(lines) <= 16:
            score += 2.0
        elif 8 <= len(lines) <= 20:
            score += 1.0
        else:
            score -= 1.0
        syllable_ok = 0
        for line in lines[:14]:
            words = line.split()
            if 8 <= len(words) <= 12:
                syllable_ok += 1
        if lines:
            ratio = syllable_ok / min(len(lines), 14)
            score += ratio * 2.0
    elif category == "math":
        for kw in ['contradict', 'assum', 'even', 'odd']:
            if re.search(kw, output, re.I):
                score += 1.0
        if re.search(r'Q\.?E\.?D|\u25a0|therefore|thus', output, re.I):
            score += 1.0
    elif category == "tool":
        for check in [bool(re.search(r'[\{\[]', output)),
                      bool(re.search(r'(search|query|tool)', output, re.I)),
                      bool(re.search(r'(argument|param|call)', output, re.I))]:
            if check:
                score += 1.0
    elif category == "prose":
        kws = re.findall(r'(thyroid|antibod|TPO|hypothyroid|euthyroid|autoimmune|TSH|lymphocyte)', output, re.I)
        if len(kws) >= 5:
            score += 2.0
        elif len(kws) >= 3:
            score += 1.0
    elif category in ("sqlite", "python", "html"):
        if re.search(r'(def |class |CREATE TABLE|<html|function )', output):
            score += 1.5
        if re.search(r'(import |from |SELECT |INSERT |<!DOCTYPE)', output):
            score += 1.0
    elif category == "email":
        if re.search(r'(Dear |Hello |Hi )', output):
            score += 1.0
        if re.search(r'(Sincerely|Best regards|Regards)', output, re.I):
            score += 1.0
        if len(output) > 200:
            score += 1.0

    words = output.lower().split()
    if len(words) > 50:
        unique = len(set(words))
        ratio = unique / len(words)
        if ratio < 0.3:
            score -= 2.0
        elif ratio < 0.5:
            score -= 1.0

    return max(0, min(10, round(score, 1)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["q4", "q5", "both"], default="both")
    args = parser.parse_args()

    # Use curated TUNED settings (derived from manual analysis of sweep data)
    # The sweep quality heuristic is too flat to auto-select; manual analysis
    # of output quality, speed, and completeness was used to pick these.
    print("Using curated tuned settings (from sweep analysis):")
    for cat, settings in TUNED.items():
        print(f"  {cat}: {settings}")

    models_to_test = ["q4", "q5"] if args.model == "both" else [args.model]
    all_results = {"meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}}

    # Load default results for comparison
    default_path = os.path.join(RESULTS_DIR, "benchmark_default.json")
    default_data = None
    if os.path.exists(default_path):
        with open(default_path) as f:
            default_data = json.load(f)

    for mk in models_to_test:
        print(f"\n{'='*70}")
        print(f"  Tuned Retest: {mk.upper()} ({os.path.basename(MODELS[mk])})")
        print(f"  Context: {CONTEXT} | Tokens: {N_TOKENS} | Threads: {THREADS}")
        print(f"{'='*70}")

        results = []
        for pid, cat, label, filename in PROMPT_DEFS:
            prompt_path = os.path.join(PROMPTS_DIR, filename)
            with open(prompt_path) as f:
                prompt_text = f.read().strip()

            settings = TUNED.get(cat, TUNED["prose"])
            print(f"\n  [{mk}] {pid}: {label} (temp={settings['temp']} top_p={settings['top_p']} top_k={settings['top_k']} rep={settings['repeat_penalty']})", end="", flush=True)

            r = run_llama_cli(MODELS[mk], prompt_text, settings)
            r["id"] = pid
            r["model"] = mk
            r["category"] = cat
            r["label"] = label
            r["settings"] = settings
            r["quality_score"] = score_quality(cat, r.get("output", ""))

            if "error" in r:
                print(f" ERROR: {r['error']}")
            else:
                print(f" q={r['quality_score']:.1f} {r['gen_tps']:.1f}t/s {r['output_chars']}ch {r['wall_time_s']}s")
            results.append(r)

        all_results[mk] = results

    # Save (merge with existing if present)
    outfile = os.path.join(RESULTS_DIR, "benchmark_tuned.json")
    existing = {}
    if os.path.exists(outfile):
        with open(outfile) as f:
            existing = json.load(f)
    existing.update(all_results)
    with open(outfile, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {outfile}")

    # Print comparison table
    print(f"\n{'='*100}")
    print(f"{'ID':<5} {'Cat':<8} {'Label':<28} {'Q4-B':>5} {'Q4-A':>5} {'dQ4':>5} {'Q5-B':>5} {'Q5-A':>5} {'dQ5':>5} {'Q4t/s':>6} {'Q5t/s':>6}")
    print(f"{'-'*100}")

    q4_def = default_data.get("q4", []) if default_data else []
    q5_def = default_data.get("q5", []) if default_data else []
    q4_tuned = all_results.get("q4", [])
    q5_tuned = all_results.get("q5", [])

    q4_scores = []
    q5_scores = []

    for pid, cat, label, _ in PROMPT_DEFS:
        d4 = next((r for r in q4_def if r["id"] == pid), {})
        d5 = next((r for r in q5_def if r["id"] == pid), {})
        t4 = next((r for r in q4_tuned if r["id"] == pid), {})
        t5 = next((r for r in q5_tuned if r["id"] == pid), {})

        q4b = score_quality(cat, d4.get("output", "")) if d4 else 0
        q4a = t4.get("quality_score", 0) if t4 else 0
        q5b = score_quality(cat, d5.get("output", "")) if d5 else 0
        q5a = t5.get("quality_score", 0) if t5 else 0

        dq4 = q4a - q4b
        dq5 = q5a - q5b
        q4tps = f"{t4['gen_tps']:.1f}" if t4 and "gen_tps" in t4 else "—"
        q5tps = f"{t5['gen_tps']:.1f}" if t5 and "gen_tps" in t5 else "—"

        if q4a:
            q4_scores.append(q4a)
        if q5a:
            q5_scores.append(q5a)

        print(f"{pid:<5} {cat:<8} {label:<28} {q4b:>5.1f} {q4a:>5.1f} {dq4:>+5.1f} {q5b:>5.1f} {q5a:>5.1f} {dq5:>+5.1f} {q4tps:>6} {q5tps:>6}")

    print(f"{'-'*100}")
    if q4_scores:
        print(f"{'AVG':<5} {'':<8} {'':<28} {'':>5} {sum(q4_scores)/len(q4_scores):>5.1f} {'':>5} {'':>5} {sum(q5_scores)/len(q5_scores):>5.1f}")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()