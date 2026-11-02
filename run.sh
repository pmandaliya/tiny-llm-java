#!/usr/bin/env bash
# Tiny helper so you only need a JDK (17+). No Maven/Gradle required.
#   ./run.sh build
#   ./run.sh gradcheck
#   ./run.sh train    [--steps 3000 --dmodel 128 ...]
#   ./run.sh generate [--prompt "ROMEO:" --tokens 400 ...]
set -euo pipefail
cd "$(dirname "$0")"

build() {
  mkdir -p target/classes
  javac --release 17 -d target/classes $(find src/main/java -name '*.java')
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  build)     build; echo "Built into target/classes" ;;
  gradcheck) build; java -cp target/classes dev.tinyllm.GradCheck ;;
  train)     build; java -cp target/classes dev.tinyllm.Train "$@" ;;
  generate)  build; java -cp target/classes dev.tinyllm.Generate "$@" ;;
  *) sed -n '2,6p' "$0" ;;
esac
