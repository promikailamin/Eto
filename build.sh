#!/usr/bin/env bash
# Build the whole project with the JDK only. No Maven, Gradle, or jars.
set -euo pipefail
cd "$(dirname "$0")"

rm -rf build
mkdir -p build
mapfile -t sources < <(find src -name '*.java' | sort)
javac --release 17 -encoding UTF-8 -d build "${sources[@]}"
jar --create --file build/eto.jar --main-class eto.cli.Main -C build .
echo "built build/eto.jar ($(du -h build/eto.jar | cut -f1))"
