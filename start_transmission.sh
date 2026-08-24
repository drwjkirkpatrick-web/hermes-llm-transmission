#!/usr/bin/env bash
# start_transmission.sh — Start the full LLM transmission stack.
#
# Starts:
#   1. llama-server (Q5 model, 32K context, Q8 KV cache) on :8080
#   2. transmission_proxy (auto-classifier + tuned params) on :8081
#
# Usage: ./start_transmission.sh
#   Stop with: ./stop_transmission.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-$HOME/models/hermes3-3b-q5_k_m.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
PORT_LLAMA=8080
PORT_PROXY=8081
CONTEXT=65536

# --- Check prerequisites ---
if [[ ! -f "$MODEL" ]]; then
  echo "Error: model not found at $MODEL" >&2
  exit 1
fi
if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "Error: llama-server not found at $LLAMA_SERVER" >&2
  exit 1
fi

# --- Stop any existing instances ---
pkill -f "llama-server.*$MODEL" 2>/dev/null || true
pkill -f "transmission_proxy.py" 2>/dev/null || true
sleep 1

# --- Start llama-server ---
echo "Starting llama-server (Q5, ${CONTEXT}K context, Q4 KV cache) on :${PORT_LLAMA}..."
"$LLAMA_SERVER" \
  -m "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT_LLAMA" \
  -c "$CONTEXT" \
  -ngl 99 \
  -t 6 \
  -fa on \
  -ctk q4_0 \
  -ctv q4_0 \
  --jinja \
  > /tmp/llama-server.log 2>&1 &

LLAMA_PID=$!
echo "  PID: $LLAMA_PID"

# --- Wait for llama-server to be ready ---
echo -n "  Waiting for llama-server..."
for i in $(seq 1 30); do
  if curl -s http://localhost:$PORT_LLAMA/health 2>/dev/null | grep -q "ok"; then
    echo " ready!"
    break
  fi
  echo -n "."
  sleep 2
  if [[ $i -eq 30 ]]; then
    echo " TIMEOUT"
    echo "Check /tmp/llama-server.log for errors"
    exit 1
  fi
done

# --- Start transmission proxy ---
echo "Starting transmission proxy on :${PORT_PROXY}..."
cd "$SCRIPT_DIR"
python3 transmission_proxy.py > /tmp/transmission-proxy.log 2>&1 &
PROXY_PID=$!
echo "  PID: $PROXY_PID"

# --- Wait for proxy ---
sleep 2
if curl -s http://localhost:$PORT_PROXY/health 2>/dev/null | grep -q "ok"; then
  echo "  Proxy ready!"
else
  echo "  Proxy failed to start - check /tmp/transmission-proxy.log"
  exit 1
fi

# --- Save PIDs ---
echo "$LLAMA_PID" > /tmp/llama-server.pid
echo "$PROXY_PID" > /tmp/transmission-proxy.pid

echo ""
echo "=== Transmission stack running ==="
echo "  llama-server:  http://localhost:$PORT_LLAMA  (PID $LLAMA_PID)"
echo "  proxy:         http://localhost:$PORT_PROXY  (PID $PROXY_PID)"
echo "  model:         hermes3-3b-q5 (Q5_K_M, 64K ctx, Q4 KV)"
echo ""
echo "  Hermes config: model.base_url = http://localhost:$PORT_PROXY/v1"
echo ""
echo "  Logs: /tmp/llama-server.log, /tmp/transmission-proxy.log"
echo "  Stop: ./stop_transmission.sh"