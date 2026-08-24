#!/usr/bin/env python3
"""
hermes_auto.py — Automatic settings adjuster for Hermes 3 3B.

Reads a prompt, classifies it, and outputs the empirically-best settings
from the variable sweep.

Classification categories (mapped from the 12 benchmark prompts):
  tool      — tool calls, JSON function calls, API interactions
  prose     — expository writing, clinical explanations, summaries
  email     — email drafts, professional correspondence
  sqlite    — SQL/SQLite code generation
  poetry    — structured poetry, iambic pentameter, sonnets
  math      — mathematical proofs, calculations, logic
  html      — HTML/CSS web page coding
  python    — Python programming

Usage:
  python3 hermes_auto.py "your prompt text"
  echo "your prompt" | python3 hermes_auto.py

Output (one line, space-separated): temp top_p top_k repeat_penalty style context tokens
  e.g.  0.7 0.90 60 1.15 tool 32768 4096
"""
import sys
import re

# ── Keyword banks ─────────────────────────────────────────────────────
TOOL_WORDS = [
    r"\btool\b", r"\btool_call\b", r"\bfunction call\b", r"\bAPI\b",
    r"\bJSON\b", r"\bsearch\b.*\bquery\b", r"\bschedule\b", r"\bcalendar\b",
    r"\bsend email\b", r"\bbook\b.*\bappointment\b",
    r"\bretrieve\b", r"\bquery\b.*\bdatabase\b",
    r"\bfile management\b", r"\borganize\b.*\bfiles\b",
    r"\brename\b.*\bfiles\b", r"\blist\b.*\bfiles\b",
    r"\btool_calls\b", r"\bfunction\b.*\bargument",
]

POETRY_WORDS = [
    r"\bpoem\b", r"\bsonnet\b", r"\bhaiku\b", r"\blimerick\b", r"\bode\b",
    r"\bvillanelle\b", r"\bacrostic\b", r"\bepigram\b",
    r"\biambic\b", r"\bpentameter\b", r"\bpentametre\b", r"\bblank verse\b",
    r"\brhyme\b", r"\brhyming\b", r"\bverse\b", r"\bstanza\b",
    r"\bmeter\b", r"\bmetre\b", r"\bscansion\b",
    r"write a poem", r"compose a poem",
]

PROSE_WORDS = [
    r"\bexplain\b", r"\bdescribe\b", r"\bsummarize\b", r"\boverview\b",
    r"\bdiscuss\b", r"\bcompare\b", r"\bcontrast\b",
    r"\bwhat is\b", r"\bwhat are\b", r"\bwhat causes\b",
    r"\bhow does\b", r"\bhow do\b",
    r"\bpathophysiology\b", r"\bdiagnosis\b", r"\btreatment\b",
    r"\bclinical\b", r"\bmedical\b", r"\bdisease\b", r"\bcondition\b",
    r"\barticle\b", r"\bessay\b", r"\breport\b", r"\bbriefing\b",
    r"\bwrite about\b", r"\bdescribe the\b",
]

EMAIL_WORDS = [
    r"\bemail\b", r"\bwrite.*email\b", r"\bemail.*draft\b",
    r"\bTo:\b", r"\bCc:\b", r"\bSubject:\b",
    r"\bcolleague\b", r"\breferral\b",
    r"\bprofessional.*tone\b", r"\bsend.*message\b",
    r"\binform.*colleague\b",
]

SQLITE_WORDS = [
    r"\bSQLite\b", r"\bsqlite\b", r"\bSQL\b.*\bquery\b",
    r"\bCREATE TABLE\b", r"\bSELECT\b.*\bFROM\b",
    r"\bdatabase\b.*\btable\b", r"\bINSERT\b",
    r"\bquery\b.*\bexecute\b", r"\bschema\b",
]

MATH_WORDS = [
    r"\bprove\b", r"\bproof\b", r"\btheorem\b", r"\blemma\b", r"\bcorollary\b",
    r"\bsolve\b", r"\bcalculate\b", r"\bcompute\b", r"\bequation\b",
    r"\bcontradiction\b", r"\binduction\b", r"\bderivation\b",
    r"\bintegral\b", r"\bderivative\b", r"\balgebra\b",
    r"\bsqrt\b", r"\bsquare root\b", r"\birrational\b",
    r"\bmatrix\b", r"\beigen\b",
]

HTML_WORDS = [
    r"\bHTML\b", r"\bhtml\b", r"\bCSS\b", r"\bcss\b",
    r"\bweb page\b", r"\bwebpage\b", r"\bweb site\b", r"\bwebsite\b",
    r"\bflexbox\b", r"\bresponsive\b", r"\bmedia query\b",
    r"\bportfolio\b", r"\blanding page\b",
    r"<!DOCTYPE", r"<html", r"<div", r"<!DOCTYPE html",
]

PYTHON_WORDS = [
    r"\bPython\b", r"\bpython\b", r"\bclass\b.*\bdef\b",
    r"\bdef\b.*\bself\b", r"\bimport\b",
    r"\btype hints\b", r"\bdocstring\b",
    r"\bdataclass\b", r"\b__main__\b",
    r"\bfunction\b.*\bPython\b", r"\bPython\b.*\bclass\b",
    r"\basync\b.*\bdef\b", r"\byield\b",
]

# ── Settings per category ─────────────────────────────────────────────
# Empirically derived from Q5_K_M variable sweep (88 runs) on Jetson.
# Categories grouped into 4 archetypes based on sweep results:
#
# REASONING (math, sqlite, html, python): low temp for determinism,
#   high top_p for vocabulary breadth, top_k=40 (sweep best for math),
#   rep=1.15 (sweep best for math, prevents loops in code/proofs).
#
# TOOL CALLS: low temp for JSON accuracy, top_p=0.95 (sweep best),
#   top_k=40 (default, no penalty), rep=1.2 (sweep best, prevents
#   repetitive JSON key patterns).
#
# PROSE (prose, email): moderate temp for natural flow,
#   top_p=0.95, top_k=40, rep=1.05 (sweep showed 1.0 too loose, 1.1 fine).
#
# CREATIVE (poetry): low-moderate temp for structure adherence,
#   top_p=0.95, top_k=80 (sweep best — more token diversity for
#   creative word choice), rep=1.2 (sweep best).
SETTINGS = {
    "tool":      {"temp": 0.1,  "top_p": 0.95, "top_k": 40,  "repeat_penalty": 1.20, "tokens": 4096},
    "prose":     {"temp": 0.8,  "top_p": 0.95, "top_k": 40,  "repeat_penalty": 1.05, "tokens": 4096},
    "email":     {"temp": 0.8,  "top_p": 0.95, "top_k": 40,  "repeat_penalty": 1.05, "tokens": 4096},
    "sqlite":    {"temp": 0.1,  "top_p": 0.98, "top_k": 40,  "repeat_penalty": 1.15, "tokens": 4096},
    "poetry":    {"temp": 0.3,  "top_p": 0.95, "top_k": 80,  "repeat_penalty": 1.20, "tokens": 4096},
    "math":      {"temp": 0.1,  "top_p": 0.98, "top_k": 40,  "repeat_penalty": 1.15, "tokens": 4096},
    "html":      {"temp": 0.1,  "top_p": 0.98, "top_k": 40,  "repeat_penalty": 1.15, "tokens": 4096},
    "python":    {"temp": 0.1,  "top_p": 0.98, "top_k": 40,  "repeat_penalty": 1.15, "tokens": 4096},
}

CONTEXT = 65536  # 64K — Hermes minimum, Q4 KV cache fits in Jetson memory

# ── Scoring ──────────────────────────────────────────────────────────
def score_category(text, patterns):
    count = 0
    t = text.lower()
    for pat in patterns:
        if re.search(pat, t):
            count += 1
    return count

def classify(prompt):
    """Classify a prompt and return (settings_dict, style_name)."""
    scores = {
        "tool":       score_category(prompt, TOOL_WORDS),
        "poetry":     score_category(prompt, POETRY_WORDS),
        "prose":      score_category(prompt, PROSE_WORDS),
        "email":      score_category(prompt, EMAIL_WORDS),
        "sqlite":     score_category(prompt, SQLITE_WORDS),
        "math":       score_category(prompt, MATH_WORDS),
        "html":       score_category(prompt, HTML_WORDS),
        "python":     score_category(prompt, PYTHON_WORDS),
    }

    # Pick highest. Ties break toward more specific categories.
    # Priority order for tie-breaking (most specific first):
    priority = ["poetry", "email", "sqlite", "html", "python", "math", "tool", "prose"]
    best_score = max(scores.values())
    if best_score == 0:
        return SETTINGS["prose"], "prose"

    for cat in priority:
        if scores[cat] == best_score:
            return SETTINGS[cat], cat

    return SETTINGS["prose"], "prose"

# ── Main ─────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    else:
        print("Usage: python3 hermes_auto.py 'your prompt'", file=sys.stderr)
        sys.exit(1)

    if not prompt:
        print("Error: empty prompt", file=sys.stderr)
        sys.exit(1)

    settings, style = classify(prompt)

    # Output: temp top_p top_k repeat_penalty style context tokens
    print(f"{settings['temp']} {settings['top_p']} {settings['top_k']} {settings['repeat_penalty']} {style} {CONTEXT} {settings['tokens']}")

if __name__ == "__main__":
    main()