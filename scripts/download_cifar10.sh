#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data}"
ARCHIVE="$DATA_ROOT/cifar-10-python.tar.gz"
URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

mkdir -p "$DATA_ROOT"
wget -O "$ARCHIVE" "$URL"
tar -xzf "$ARCHIVE" -C "$DATA_ROOT"
echo "CIFAR-10 is ready under $DATA_ROOT/cifar-10-batches-py"
