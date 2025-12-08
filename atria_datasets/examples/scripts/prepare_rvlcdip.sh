#!/bin/bash

python atria_datasets/src/atria_datasets/prepare_dataset.py rvlcdip/image_with_ocr --max-samples 100
python atria_datasets/src/atria_datasets/prepare_dataset.py tobacco3482/image_with_ocr --max-samples 100