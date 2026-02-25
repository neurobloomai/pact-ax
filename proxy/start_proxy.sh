#!/usr/bin/env bash
# start_proxy.sh — Run PACT-AX proxy standalone (outside Cursor)
# Useful for smoke-testing the proxy before wiring it into mcp.json.
#
# Usage:
#   export GITHUB_PERSONAL_ACCESS_TOKEN=ghp_xxxx
#   ./start_proxy.sh
#
# Optional overrides:
#   PACT_UPSTREAM_MODE=docker   (default) | npx
#   PACT_DRIFT_THRESHOLD=0.3    float
#   PACT_BLOCK_ON_VIOLATION=false | true

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${GITHUB_PERSONAL_ACCESS_TOKEN:-}" ]]; then
  echo "❌  GITHUB_PERSONAL_ACCESS_TOKEN is not set."
  echo "    Export it first:  export GITHUB_PERSONAL_ACCESS_TOKEN=ghp_xxxx"
  exit 1
fi

# Verify Docker is available (required for default upstream mode)
UPSTREAM_MODE="${PACT_UPSTREAM_MODE:-docker}"
if [[ "$UPSTREAM_MODE" == "docker" ]]; then
  if ! command -v docker &>/dev/null; then
    echo "❌  Docker not found. Install Docker Desktop or set PACT_UPSTREAM_MODE=npx"
    exit 1
  fi
  if ! docker info &>/dev/null; then
    echo "❌  Docker daemon is not running. Start Docker Desktop first."
    exit 1
  fi
  # Pull image if needed (silent if already present)
  echo "🐳  Pulling GitHub MCP server image (if needed)..."
  docker pull ghcr.io/github/github-mcp-server --quiet
fi

echo "🚀  Starting PACT-AX proxy  (upstream=$UPSTREAM_MODE)"
echo "    Logging to stderr. MCP messages on stdin/stdout."
echo "    Press Ctrl-C to stop."
echo ""

cd "$REPO_ROOT"
python -m proxy.src.proxy
