#!/usr/bin/env bash
# Everything on all three models (seeds 1-5). Same as: bash scripts/submit_lethe_compare.sh full all
exec bash "$(dirname "$0")/submit_lethe_compare.sh" full "${@:-all}"
