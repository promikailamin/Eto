#!/usr/bin/env bash
# Train your own GGUF model. Pass --help for all options.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f build/eto.jar ]; then
  ./build.sh
fi
exec java -Xmx3g -cp build eto.cli.Main train "$@"
