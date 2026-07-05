#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

declare -a datasets=(
    "cifar10 --config-name 1k-75baa466 --download-dir data/"
    "tobacco3482 --config-name image_with_ocr-1f5da6de --download-dir data/"
    "cord --download-dir data/" # config-name will be resolved
    "funsd --download-dir data/" # config-name will be resolved
    "sroie --download-dir data/" # config-name will be resolved
    "wild-receipts --download-dir data/" # config-name will be resolved
    "fintabnet --download-dir data/" 
    "icdar2019 --config-name trackA_modern --download-dir data/"
    "due-benchmark --config-name DocVQA --download-dir data/"  
)


for dataset_entry in "${datasets[@]}"; do
    echo "Downloading dataset: $dataset_entry"
    uv run atria datasets download $dataset_entry $@
done
