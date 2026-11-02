#!/usr/bin/env bash
# Ollama-style CLI chat runner. Works with this project's GGUF and with
# third-party llama / mistral / qwen2 GGUF models.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f build/eto.jar ]; then
  ./build.sh
fi
HEAP="${ETO_HEAP:-4g}"
OPTS=(-Xmx"$HEAP")
if [ -n "${ETO_THREADS:-}" ]; then
  OPTS+=("-Deto.threads=$ETO_THREADS")
fi
exec java "${OPTS[@]}" -cp build eto.cli.Main chat "$@"
