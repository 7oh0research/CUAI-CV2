#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."; python scripts/analyze_h3.py
