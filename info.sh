#!/usr/bin/env bash
# Inspect GGUF metadata, tokenizer and tensor types.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f build/eto.jar ]; then
  ./build.sh
fi
exec java -Xmx1g -cp build eto.cli.Main info "$@"
