#!/usr/bin/env bash
set -euo pipefail

PACKAGE=$1

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==============================${NC}"
echo -e "${BLUE} Running type checks with mypy ${NC}"
echo -e "${BLUE}==============================${NC}"
./ci/lint.sh $PACKAGE

echo -e "\n${BLUE}==============================${NC}"
echo -e "${BLUE} Running linter with Ruff     ${NC}"
echo -e "${BLUE}==============================${NC}"
./ci/format.sh $PACKAGE

echo -e "\n${BLUE}==============================${NC}"
echo -e "${BLUE} Running tests with pytest    ${NC}"
echo -e "${BLUE}==============================${NC}"
./ci/test.sh $PACKAGE

echo -e "\n${GREEN}=== All checks passed! ✅ ===${NC}"
