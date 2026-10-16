#!/bin/bash

# Get the absolute path of the test directory
TEST_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "Checking for clean.sh in test subdirectories..."

# Iterate through all subdirectories in the test folder
for dir in "$TEST_DIR"/*/; do
    # Skip directories that don't have a clean.sh file
    if [ -f "$dir/clean.sh" ]; then
        dirname=$(basename "$dir")
        
        # Ask for confirmation
        read -p "Run clean.sh in '$dirname'? (y/N): " confirm
        
        if [[ "$confirm" =~ ^[yY](es)?$ ]]; then
            echo "--> Cleaning '$dirname'..."
            # Use subshell to cd and run the script without affecting current process
            (cd "$dir" && bash clean.sh)
        else
            echo "--> Skipping '$dirname'."
        fi
    fi
done

echo "Done."
