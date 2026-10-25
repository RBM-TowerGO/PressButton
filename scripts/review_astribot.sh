#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/.venv-data-review/bin/python" "$ROOT/scripts/astribot_data_review.py" "$@"
