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

    echo "Formatting package: $package"
    uv run ruff check $package/src --fix      # linter
    uv run ruff format $package/src # formatter
done
