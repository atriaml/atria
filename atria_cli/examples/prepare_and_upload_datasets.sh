#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

declare -a datasets=(
    "cifar10/1k"
    # "tobacco3482/image_with_ocr"
    # "cord"
    # "funsd"
    # "sroie"
    # "wild_receipts"
    # "fintabnet/1k --max_train_samples 100 --max_validation_samples 100 --max_test_samples 100" 
    # "icdar2019/trackA_modern"
    # "due_benchmark/DocVQA --max_train_samples 100 --max_validation_samples 100 --max_test_samples 100"  
)


for dataset_entry in "${datasets[@]}"; do
    echo "Processing dataset: $dataset_entry"
    python -m atria_cli.cli datasets prepare_and_upload $dataset_entry $@ 
done