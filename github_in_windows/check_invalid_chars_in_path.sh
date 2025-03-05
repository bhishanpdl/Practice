#!/bin/bash

# List of invalid characters for Windows file names
invalid_chars='[\|\?\*\:"]'

# Function to check for invalid characters in file paths using git ls-tree
check_invalid_chars() {
    git ls-tree -r HEAD --name-only | grep -E "$invalid_chars"
}

# Run the check and display results
invalid_files=$(check_invalid_chars)

if [ -n "$invalid_files" ]; then
    echo "Files with invalid characters in their paths:"
    echo "$invalid_files"
else
    echo "No files with invalid characters found."
fi
