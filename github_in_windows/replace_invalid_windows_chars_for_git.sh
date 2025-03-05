#!/bin/bash

# Set the base directory
base=~/github/Entertainment

# List of invalid characters for Windows file names
invalid_chars='[\|\?\*\:"]'

# Function to check for invalid characters in file paths using git ls-tree
check_invalid_chars() {
    git -C "$base" ls-tree -r HEAD --name-only | grep -E "$invalid_chars"
}

# Function to replace invalid characters with underscores and rename files
replace_invalid_chars() {
    files_with_invalid_chars=$(check_invalid_chars)
    for file in $files_with_invalid_chars; do
        new_file=$(echo "$file" | sed 's/[\|\?\*\:"]/ /g')
        echo "Renaming: $file -> $new_file"
        git -C "$base" mv "$file" "$new_file"
    done
}

# Navigate to the base directory and run the check
replace_invalid_chars

echo "Done replacing invalid characters and renaming files."
