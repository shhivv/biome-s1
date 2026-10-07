#!/usr/bin/env bash
# Publish the exported model to the Hugging Face Hub.
#   ./scripts/publish_hf.sh <user-or-org>/<repo> [--private] [release dir, default release/hf]
#   ./scripts/publish_hf.sh shhivv/mesa-s1 --private release/mesa-s1   (public: pass "" for --private)
# Requires `hf auth login` first (token with write access).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=${1:?usage: publish_hf.sh <user>/<repo> [--private]}
VIS=${2:-}
DIR=${3:-release/hf}
NAME=$(python3 -c "import json; m=json.load(open('$DIR/config.json'))['metadata']; print(m.get('name','model'), m.get('version',''))")
sed -i '' "s|HF_REPO_ID|$REPO|g" "$DIR/README.md"
.venv/bin/hf repos create "$REPO" --repo-type model $VIS --exist-ok
.venv/bin/hf upload "$REPO" "$DIR" . --repo-type model --commit-message "$NAME"
echo "https://huggingface.co/$REPO"
