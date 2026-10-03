#!/usr/bin/env bash
# Downloads the AOM AV1 test vectors (bitstreams plus the MD5 of every
# decoded frame) into tests/vectors/. They are data published by AOMedia
# for decoder testing:
#   https://storage.googleapis.com/aom-test-data/
# The list in tools/vectors.txt is every av1-1-b8-* and av1-1-b10-* stream
# in that bucket (244 streams, about 7 MB).
set -euo pipefail
cd "$(dirname "$0")/.."
base=https://storage.googleapis.com/aom-test-data
mkdir -p tests/vectors
while read -r name; do
  [ -z "$name" ] && continue
  for f in "$name" "$name.md5"; do
    [ -s "tests/vectors/$f" ] || curl -sSfL --retry 3 -o "tests/vectors/$f" "$base/$f"
  done
done < tools/vectors.txt
echo "tests/vectors: $(ls tests/vectors | grep -vc '\.md5$') streams"
