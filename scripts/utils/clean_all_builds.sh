#!/bin/bash
# Recursive Build Cleaner Wrapper
# Usage: ./clean_all_builds.sh <search_path> [--dry-run]

DRY_RUN_FLAG=""
if [[ "$*" == *"--dry-run"* ]]; then
    DRY_RUN_FLAG="--dry-run"
fi

SEARCH_PATH=$1
if [ -z "$SEARCH_PATH" ] || [ "$SEARCH_PATH" == "--dry-run" ]; then
    echo "Usage: $0 <search_path> [--dry-run]"
    exit 1
fi

# Get the absolute path of the cleaner script
CLEANER_SCRIPT="$(dirname "$(readlink -f "$0")")/build_cleaner.py"

echo "Searching for 'build' directories in: $SEARCH_PATH"
if [ "$DRY_RUN_FLAG" == "--dry-run" ]; then
    echo "Mode: DRY-RUN (No files will be deleted)"
fi

# Find all directories named 'build' and run the cleaner on them
find "$SEARCH_PATH" -name "build" -type d | while read -r build_dir; do
    echo "----------------------------------------------------------------"
    echo "Cleaning: $build_dir"
    python3 "$CLEANER_SCRIPT" "$build_dir" $DRY_RUN_FLAG
done

echo "----------------------------------------------------------------"
echo "Done."
