#!/bin/bash

PACKAGES=("atria_datasets" "atria_transforms" "atria_models" "atria_ml" "atria_metrics")

TARGET_PACKAGE=$1

if [ -n "$TARGET_PACKAGE" ]; then
    PACKAGES=("$TARGET_PACKAGE")
fi

for package in "${PACKAGES[@]}"; do
    echo "Building registry for ${package}..."
    ATRIA_LOG_LEVEL=DEBUG BUILD_REGISTRY=1 uv run python "${package}/src/${package}/build_registry.py"
    if [ $? -eq 0 ]; then
        echo "✓ Successfully built registry for ${package}"
    else
        echo "✗ Failed to build registry for ${package}"
        exit 1
    fi
done

echo "All registries built successfully"