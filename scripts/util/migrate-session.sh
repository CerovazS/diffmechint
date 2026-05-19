#!/usr/bin/env bash
# Migrate a Claude Code session + this repo to another machine.
#
# Copies (a) the project tree (excluding .venv / caches / outputs),
# (b) the session JSONL and subagent transcript dir, (c) the per-project
# memory dir. Renames the project hash directory so the target can resume
# from its own working directory layout.
#
# Usage:
#   scripts/migrate-session.sh <SSH_HOST> <SESSION_ID> [TARGET_PROJECT_PATH]
#
# Example:
#   scripts/migrate-session.sh 100.124.107.92 \
#     0e2cf8f0-49fe-4176-9bd8-252e9d9ed938 \
#     /home/cerovaz/repos/diffmechint

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <SSH_HOST> <SESSION_ID> [TARGET_PROJECT_PATH]" >&2
    exit 2
fi

SSH_HOST="$1"
SESSION_ID="$2"
TARGET_PROJECT="${3:-/home/${SSH_HOST##*@}/repos/diffmechint}"

# Resolve the source project hash from PWD: replace '/' with '-'.
SRC_PROJECT_PATH="$(realpath ~)"
SRC_HASH="$(echo "$SRC_PROJECT_PATH" | tr '/' '-')"
SRC_SESSION_ROOT="$HOME/.claude/projects/${SRC_HASH}"

if [[ ! -f "${SRC_SESSION_ROOT}/${SESSION_ID}.jsonl" ]]; then
    echo "Session ${SESSION_ID} not found at ${SRC_SESSION_ROOT}" >&2
    exit 3
fi

# Target hash from TARGET_PROJECT. The launch CWD on the target is the
# parent of TARGET_PROJECT (so /home/$user is the launch dir for /home/$user/repos/diffmechint).
# But common practice is to launch from the project itself, so we encode TARGET_PROJECT directly.
TGT_HASH="$(echo "$TARGET_PROJECT" | tr '/' '-')"
TGT_SESSION_ROOT=".claude/projects/${TGT_HASH}"

echo "[migrate] source project hash: ${SRC_HASH}"
echo "[migrate] target host:         ${SSH_HOST}"
echo "[migrate] target project:      ${TARGET_PROJECT}"
echo "[migrate] target session root: ~/${TGT_SESSION_ROOT}"

ssh "$SSH_HOST" "mkdir -p '${TARGET_PROJECT}' '~/${TGT_SESSION_ROOT}'"

PROJ_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
echo "[migrate] rsync project ${PROJ_ROOT} -> ${SSH_HOST}:${TARGET_PROJECT}"
rsync -a --info=progress2 \
    --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
    --exclude='.ruff_cache' --exclude='.mypy_cache' \
    --exclude='outputs' --exclude='hf_cache' \
    "${PROJ_ROOT}/" "${SSH_HOST}:${TARGET_PROJECT}/"

echo "[migrate] rsync session jsonl"
rsync -a "${SRC_SESSION_ROOT}/${SESSION_ID}.jsonl" \
    "${SSH_HOST}:~/${TGT_SESSION_ROOT}/"

if [[ -d "${SRC_SESSION_ROOT}/${SESSION_ID}" ]]; then
    echo "[migrate] rsync session subagent dir"
    rsync -a "${SRC_SESSION_ROOT}/${SESSION_ID}/" \
        "${SSH_HOST}:~/${TGT_SESSION_ROOT}/${SESSION_ID}/"
fi

if [[ -d "${SRC_SESSION_ROOT}/memory" ]]; then
    echo "[migrate] rsync project memory"
    rsync -a "${SRC_SESSION_ROOT}/memory/" \
        "${SSH_HOST}:~/${TGT_SESSION_ROOT}/memory/"
fi

cat <<EOF

[migrate] DONE.

On ${SSH_HOST}:
    cd ${TARGET_PROJECT}
    uv sync             # first time only — rebuilds .venv
    claude --resume ${SESSION_ID}

Note: MCP server auth (Notion, Flywheel, etc.) is machine-local and must be
re-authenticated on the target. Tool outputs in /tmp/claude-*/ are ephemeral
and do not migrate.
EOF
