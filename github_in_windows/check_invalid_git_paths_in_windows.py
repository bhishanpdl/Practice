import subprocess
import re

# List of invalid characters for Windows file names
invalid_chars = re.compile(r'[\|\?\*\:"]')

def check_invalid_chars():
    # Run git ls-tree and capture the output
    result = subprocess.run(['git', 'ls-tree', '-r', 'HEAD', '--name-only'], capture_output=True, text=True)
    files = result.stdout.splitlines()

    # Filter files with invalid characters
    invalid_files = [file for file in files if invalid_chars.search(file)]

    return invalid_files

# Run the check and display results
invalid_files = check_invalid_chars()

if invalid_files:
    print("Files with invalid characters in their paths:")
    for file in invalid_files:
        print(file)
else:
    print("No files with invalid characters found.")
