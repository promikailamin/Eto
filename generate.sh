#!/usr/bin/env bash
# One-shot generation. Pass --help for all options.
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
exec java "${OPTS[@]}" -cp build eto.cli.Main generate "$@"
