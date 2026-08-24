#!/usr/bin/env bash
# ask-hermes3.sh — Hermes 3 3B with AUTOMATIC settings adjustment.
#
# Reads your prompt, classifies it (tool/prose/email/sqlite/poetry/math/html/python),
# and picks the empirically-best temperature AND sampling parameters from
# our variable sweep on this Jetson.
#
# Context is ALWAYS 32768. GUI should be off for RAM headroom.
#
# Usage:
#   ./ask-hermes3.sh "your question or prompt here"
#   echo "your prompt" | ./ask-hermes3.sh
#   PROMPT_FILE=/path/prompt.txt ./ask-hermes3.sh
#
# Options:
#   --model q5    Use Q5_K_M (default, better quality)
#   --model q4    Use Q4_K_M (smaller, slightly faster)
#   --raw         Skip auto-classifier, use defaults (temp=0.8 top_p=0.95)
set -euo pipefail

# --- Defaults ---
MODEL_KEY="${MODEL_KEY:-q5}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
RAW_MODE=0

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_KEY="$2"
      shift 2
      ;;
    --model=*)
      MODEL_KEY="${1#*=}"
      shift
      ;;
    --raw)
      RAW_MODE=1
      shift
      ;;
    *)
      break
      ;;
  esac
done

# --- Model path ---
case "$MODEL_KEY" in
  q4) MODEL="$MODEL_DIR/hermes3-3b-q4_k_m.gguf" ;;
  q5) MODEL="$MODEL_DIR/hermes3-3b-q5_k_m.gguf" ;;
  *)  echo "Error: unknown model key '$MODEL_KEY' (use q4 or q5)" >&2; exit 1 ;;
esac

if [[ ! -f "$MODEL" ]]; then
  echo "Error: model not found at $MODEL" >&2
  exit 1
fi

# --- llama-cli path ---
LLAMA_CLI="${LLAMA_CLI:-$HOME/llama.cpp/build/bin/llama-cli}"
if [[ ! -x "$LLAMA_CLI" ]]; then
  echo "Error: llama-cli not found at $LLAMA_CLI" >&2
  exit 1
fi

# --- Prompt: arg, file, or stdin ---
if [[ $# -gt 0 ]]; then
  PROMPT="$*"
elif [[ -n "${PROMPT_FILE:-}" ]]; then
  PROMPT="$(cat "$PROMPT_FILE")"
elif [[ ! -t 0 ]]; then
  PROMPT="$(cat)"
else
  echo "Usage: $0 'your question or prompt'" >&2
  echo "   or: echo 'prompt' | $0" >&2
  echo "   or: PROMPT_FILE=/path/prompt.txt $0" >&2
  echo "Options: --model q4|q5  --raw" >&2
  exit 1
fi

if [[ -z "$PROMPT" ]]; then
  echo "Error: empty prompt" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT=32768

if [[ "$RAW_MODE" -eq 1 ]]; then
  TEMP="0.8"; TOP_P="0.95"; TOP_K="40"; REPEAT="1.1"; N_TOKENS="4096"
  STYLE="raw"
else
  # --- Auto-classify ---
  AUTO_OUTPUT=$(python3 "$SCRIPT_DIR/hermes_auto.py" "$PROMPT" 2>/dev/null || echo "")
  if [[ -z "$AUTO_OUTPUT" ]]; then
    echo "Warning: hermes_auto.py failed, falling back to defaults" >&2
    TEMP="0.8"; TOP_P="0.95"; TOP_K="40"; REPEAT="1.1"; N_TOKENS="4096"; STYLE="default"
  else
    read -r TEMP TOP_P TOP_K REPEAT STYLE CTX N_TOKENS <<< "$AUTO_OUTPUT"
  fi
fi

# --- Show settings ---
echo "== ask-hermes3.sh (auto-tuned) ==" >&2
echo "  model:     $MODEL_KEY ($(basename "$MODEL"))" >&2
echo "  prompt:    ${PROMPT:0:80}..." >&2
echo "  style:     $STYLE" >&2
echo "  temp:      $TEMP" >&2
echo "  context:   $CONTEXT" >&2
echo "  tokens:    $N_TOKENS" >&2
echo "  top_p:     $TOP_P  |  top_k: $TOP_K  |  repeat: $REPEAT" >&2
echo "=================================" >&2

# --- System prompt (Hermes 3 ChatML) ---
SYSTEM="You are Hermes 3, a helpful and knowledgeable AI assistant. You follow instructions carefully and provide accurate, well-structured responses."

# --- Run ---
CHATML="<|im_start|>system
${SYSTEM}<|im_end|>
<|im_start|>user
${PROMPT}<|im_end|>
<|im_start|>assistant
"

exec "$LLAMA_CLI" \
  -m "$MODEL" \
  -p "$CHATML" \
  -n "$N_TOKENS" \
  -c "$CONTEXT" \
  --temp "$TEMP" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --repeat-penalty "$REPEAT" \
  -t 6 \
  -ngl 99 \
  -fa on \
  --jinja \
  --no-conversation \
  --no-display-prompt \
  -st