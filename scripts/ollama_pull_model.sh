#!/bin/bash
# =============================================================================
# Ollama Model Pull Script
# Runs inside the ollama-pull-model container.
# Waits for Ollama server to be ready, then pulls the configured model.
# =============================================================================

set -e

MODEL="${OLLAMA_MODEL:-qwen2.5:0.5b}"
OLLAMA_URL="${OLLAMA_HOST:-http://ollama:11434}"
MAX_WAIT=120  # seconds to wait for Ollama to be ready

echo "=========================================="
echo "  Ollama Model Puller"
echo "  Model: $MODEL"
echo "  Server: $OLLAMA_URL"
echo "=========================================="

# Wait for Ollama server to be ready
echo "⏳ Waiting for Ollama server..."
waited=0
until curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; do
    if [ $waited -ge $MAX_WAIT ]; then
        echo "❌ Timeout waiting for Ollama after ${MAX_WAIT}s"
        exit 1
    fi
    sleep 2
    waited=$((waited + 2))
    echo "   Still waiting... (${waited}s)"
done

echo "✅ Ollama server is ready"

# Check if model is already downloaded
echo "🔍 Checking if model '$MODEL' is already available..."
if curl -sf "${OLLAMA_URL}/api/tags" | grep -q "\"$MODEL\""; then
    echo "✅ Model '$MODEL' already downloaded — skipping pull"
else
    echo "⬇️  Pulling model '$MODEL' (this may take a few minutes on first run)..."
    curl -s "${OLLAMA_URL}/api/pull" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"$MODEL\"}" \
        --no-buffer | while IFS= read -r line; do
            # Print progress lines
            status=$(echo "$line" | grep -o '"status":"[^"]*"' | sed 's/"status":"//;s/"//')
            if [ -n "$status" ]; then
                echo "   $status"
            fi
        done
    echo "✅ Model '$MODEL' ready"
fi

echo "=========================================="
echo "  Ollama is ready for use!"
echo "  API: ${OLLAMA_URL}"
echo "  Model: $MODEL"
echo "=========================================="
