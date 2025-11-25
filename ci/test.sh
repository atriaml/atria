#!/usr/bin/env bash

PACKAGE=$1

declare -a packages=(
    atria_logger
    atria_types
)

for package in "${packages[@]}"; do
    if [[ -n "$PACKAGE" && "$PACKAGE" != "$package" ]]; then
        continue
    fi

    echo "Running tests for $package..."
    uv run coverage run --source="$package" -m pytest "$@"
    uv run coverage report --show-missing
    echo "Completed tests for $package"
    echo "----------------------------------------"
done
