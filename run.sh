#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run.sh agent "<bluesky_url>"
#   ./run.sh eval

CMD="${1:-}"

if [[ "$CMD" == "agent" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "Usage: ./run.sh agent \"<bluesky_url>\""
    exit 1
  fi
  .venv/bin/python3 -m agent.main "$2"

elif [[ "$CMD" == "eval" ]]; then
  .venv/bin/python3 -m evals.run_eval

else
  echo "Usage:"
  echo "  ./run.sh agent \"<bluesky_url>\"   # explain a post"
  echo "  ./run.sh eval                     # run eval harness"
  exit 1
fi
