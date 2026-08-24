#!/usr/bin/env python3
"""
transmission_proxy.py — LLM transmission proxy for Hermes Agent.

Sits between Hermes and llama-server. For each /v1/chat/completions request:
  1. Extracts the last user message
  2. Classifies it via hermes_auto.py (tool/prose/email/sqlite/poetry/math/html/python)
  3. Injects the empirically-best sampling parameters for that category
  4. Forwards to llama-server (streaming pass-through)
  5. Returns the response

Hermes config points at this proxy instead of a cloud provider.

Usage:
  python3 transmission_proxy.py [--port 8081] [--upstream http://localhost:8080]

Requires: flask, requests
"""
import sys
import os
import json
import time
import logging
import requests

# Import the classifier from hermes_auto.py (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hermes_auto import classify, CONTEXT

from flask import Flask, request, Response, jsonify

# ── Config ────────────────────────────────────────────────────────────
UPSTREAM = os.environ.get("TRANSMISSION_UPSTREAM", "http://localhost:8080")
PORT = int(os.environ.get("TRANSMISSION_PORT", "8081"))
MODEL_NAME = "hermes3-3b-q5"

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("transmission")

# ── Helpers ────────────────────────────────────────────────────────────
def extract_last_user_message(body: dict) -> str:
    """Pull the last user message from a chat completions request body."""
    messages = body.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                # Some APIs send content as parts
                return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            return content
    return ""

def apply_tuned_params(body: dict) -> dict:
    """Classify the prompt and inject tuned sampling params."""
    prompt_text = extract_last_user_message(body)

    settings, style = classify(prompt_text)

    # Override sampling params with tuned values
    body["temperature"] = settings["temp"]
    body["top_p"] = settings["top_p"]
    body["top_k"] = settings["top_k"]
    body["repeat_penalty"] = settings["repeat_penalty"]

    # Log the routing decision
    log.info(
        "category=%s | temp=%.2f top_p=%.2f top_k=%d rep=%.2f | prompt=%s",
        style,
        settings["temp"],
        settings["top_p"],
        settings["top_k"],
        settings["repeat_penalty"],
        prompt_text[:80].replace("\n", " "),
    )

    return body

# ── Routes ──────────────────────────────────────────────────────────────
@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions endpoint with auto-tuning."""
    body = request.get_json(force=True)

    # Inject tuned params
    body = apply_tuned_params(body)

    # Ensure model name matches what llama-server expects
    body["model"] = body.get("model", MODEL_NAME)

    stream = body.get("stream", False)

    # Forward to llama-server
    upstream_url = f"{UPSTREAM}/v1/chat/completions"

    if stream:
        # Streaming pass-through
        resp = requests.post(upstream_url, json=body, stream=True, timeout=300)
        if resp.status_code != 200:
            return jsonify({"error": resp.text}), resp.status_code

        def generate():
            for chunk in resp.iter_content(chunk_size=None):
                if chunk:
                    yield chunk
        return Response(generate(), content_type="text/event-stream")
    else:
        # Non-streaming
        resp = requests.post(upstream_url, json=body, timeout=300)
        if resp.status_code != 200:
            return jsonify({"error": resp.text}), resp.status_code
        return Response(resp.content, content_type="application/json")

@app.route("/v1/models", methods=["GET"])
def list_models():
    """Fake the models list so Hermes sees a valid model."""
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    })

@app.route("/v1/completions", methods=["POST"])
def completions():
    """Plain completions endpoint (non-chat) — also tuned."""
    body = request.get_json(force=True)
    # For completions, classify the raw prompt
    prompt_text = body.get("prompt", "")
    if isinstance(prompt_text, list):
        prompt_text = " ".join(prompt_text)

    settings, style = classify(prompt_text)
    body["temperature"] = settings["temp"]
    body["top_p"] = settings["top_p"]
    body["top_k"] = settings["top_k"]
    body["repeat_penalty"] = settings["repeat_penalty"]

    log.info("completions category=%s | prompt=%s", style, prompt_text[:80])

    stream = body.get("stream", False)
    upstream_url = f"{UPSTREAM}/v1/completions"

    if stream:
        resp = requests.post(upstream_url, json=body, stream=True, timeout=300)
        if resp.status_code != 200:
            return jsonify({"error": resp.text}), resp.status_code
        def generate():
            for chunk in resp.iter_content(chunk_size=None):
                if chunk:
                    yield chunk
        return Response(generate(), content_type="text/event-stream")
    else:
        resp = requests.post(upstream_url, json=body, timeout=300)
        if resp.status_code != 200:
            return jsonify({"error": resp.text}), resp.status_code
        return Response(resp.content, content_type="application/json")

@app.route("/health", methods=["GET"])
def health():
    """Health check."""
    try:
        r = requests.get(f"{UPSTREAM}/health", timeout=5)
        upstream_ok = r.status_code == 200
    except Exception:
        upstream_ok = False
    return jsonify({
        "status": "ok" if upstream_ok else "degraded",
        "upstream": UPSTREAM,
        "upstream_ok": upstream_ok,
        "port": PORT,
    })

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "hermes-llm-transmission-proxy",
        "upstream": UPSTREAM,
        "model": MODEL_NAME,
        "endpoints": ["/v1/chat/completions", "/v1/models", "/v1/completions", "/health"],
    })

if __name__ == "__main__":
    log.info("Starting transmission proxy on :%d -> %s", PORT, UPSTREAM)
    app.run(host="0.0.0.0", port=PORT, threaded=True)