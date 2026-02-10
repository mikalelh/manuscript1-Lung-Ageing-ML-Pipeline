#!/bin/bash
# Run this in a normal terminal (NOT Cursor) so git commit works.
# Fixes: GitHub rejects push because large .h5ad files are in commit history.
# Works with older Git (no --show-current).
set -e
cd "$(dirname "$0")"

# Current branch (works on Git < 2.22)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $BRANCH"

if [ "$BRANCH" = "main" ]; then
  echo "Creating new branch 'newmain' with no history (large files will be dropped)..."
  git checkout --orphan newmain main
  echo "Creating single initial commit..."
  git commit -m "Initial commit: Ageing in the Lung ML pipeline"
  echo "Replacing main with this history..."
  git branch -D main
  git branch -m main
  echo "Pushing (force, to replace remote history)..."
  git push -f origin main
  echo "Done."
else
  echo "Please run from branch main:  git checkout main"
  echo "Then run:  ./fix_push_no_large_files.sh"
fi
