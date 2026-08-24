#!/usr/bin/env bash
# stop_transmission.sh — Stop the LLM transmission stack.
set -euo pipefail

# Kill by PID files first
for pidfile in /tmp/llama-server.pid /tmp/transmission-proxy.pid; do
  if [[ -f "$pidfile" ]]; then
    PID=$(cat "$pidfile")
    kill "$PID" 2>/dev/null && echo "Stopped PID $PID ($(basename "$pidfile" .pid))"
    rm -f "$pidfile"
  fi
done

# Fallback: kill by pattern
pkill -f "transmission_proxy.py" 2>/dev/null && echo "Stopped transmission_proxy" || true
pkill -f "llama-server.*hermes3" 2>/dev/null && echo "Stopped llama-server" || true

sleep 1
echo "Transmission stack stopped."