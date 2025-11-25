#!/usr/bin/env bash

PACKAGE=$1

source ./ci/packages.sh

for package in "${packages[@]}"; do
    if [[ -n "$PACKAGE" && "$PACKAGE" != "$package" ]]; then
        continue
    fi

    echo "Linting package: $package"
    uv run mypy $package/src --follow-imports=skip     # type check
    uv run ruff check $package/src     # linter
    uv run ruff format $package/src --check # formatter
done 