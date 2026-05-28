#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

declare -a small_datasets=(
    "cifar10/1k"
    # "huggingface_cifar10/plain_text_1k"
    "tobacco3482/image_with_ocr"
    # "rvlcdip/image_with_ocr_1k"
    # "mnist/mnist_1k"
    "cord"
    "funsd"
    # "sroie/default"
    # "wild_receipts/default"
    # "docile/kile"
    # "docbank/1k"  # too big, failing downloads
    # "fintabnet/1k" 
    # "icdar2019/trackA_modern"
    # "icdar2013/default"
    # "docvqa/default" 
)


declare -a big_datasets=(
    "rvlcdip/image_with_ocr"
    "docbank/default"  
    "doclaynet/2022.08" # too big, failing downloads
    "publaynet/default" # too big, failing downloads
    "pubtables1m/detection_1k" # too big, failing downloads
    "pubtables1m/structure_1k" # too big, failing downloads
)


if [[ "$1" == "small" ]]; then
    for dataset_entry in "${small_datasets[@]}"; do
        echo "Processing dataset: $dataset_entry"
        python -m atria_cli.cli datasets prepare_and_upload $dataset_entry ${@:1}
    done
elif [[ "$1" == "big" ]]; then
    for dataset_entry in "${big_datasets[@]}"; do
        echo "Processing dataset: $dataset_entry"
        python -m atria_cli.cli datasets prepare_and_upload $dataset_entry ${@:1}
    done
else
    python -m atria_cli.cli datasets prepare_and_upload $@
fi