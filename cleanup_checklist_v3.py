import re
import os
import sys

# Ensure we are in the right directory
if not os.path.exists("tinyops"):
    print("Error: tinyops directory not found. Must run from root.")
    sys.exit(1)

def normalize_path(path_str):
    # path_str is like "tinyops.linalg.dot"
    # convert to "tinyops/linalg/dot.py"
    return path_str.replace('.', '/') + '.py'

def check_implementation_exists(path_str):
    file_path = normalize_path(path_str)

    # Debug print for first few items
    # print(f"Checking {file_path}...")

    if os.path.exists(file_path):
        if os.path.getsize(file_path) > 50:
            return True

    # Check if it's a module
    module_path = path_str.replace('.', '/')
    if os.path.isdir(module_path) and os.path.exists(os.path.join(module_path, '__init__.py')):
         return True

    return False

def clean_line(line):
    parts = line.split('→')
    if len(parts) != 2:
        return None, None, None

    prefix = parts[0]
    rest = parts[1]

    # Remove suffixes
    rest = re.sub(r'\s*\(.*?\)\s*$', '', rest)
    rest = rest.strip()

    path_match = re.search(r'`([^`]+)`', rest)
    if path_match:
        path = path_match.group(1)
    else:
        path = rest

    return prefix, rest, path

def main():
    with open('CHECKLIST.md', 'r') as f:
        lines = f.readlines()

    unique_entries = {}
    header = []

    # Pass 1: Gather valid entries and check existence
    for line in lines:
        line = line.strip()
        if not line.startswith('- ['):
            header.append(line)
            continue

        prefix, rest, path = clean_line(line)
        if path is None:
            header.append(line)
            continue

        desc_match = re.search(r'\[(.)\] (.*)', prefix)
        if not desc_match:
             continue

        desc = desc_match.group(2).strip()
        exists = check_implementation_exists(path)

        # Deduplication key: path
        unique_entries[path] = {
            'desc': desc,
            'rest': rest,
            'checked': exists
        }

    # Pass 2: Write back
    seen_paths = set()
    output_lines = []

    for line in header:
        if line: output_lines.append(line)

    for line in lines:
        line = line.strip()
        if not line.startswith('- ['):
            continue

        prefix, rest, path = clean_line(line)
        if path is None or path in seen_paths:
            continue

        seen_paths.add(path)
        entry = unique_entries.get(path)
        if not entry: continue

        check_mark = 'x' if entry['checked'] else ' '
        output_lines.append(f"- [{check_mark}] {entry['desc']} → {entry['rest']}")

    with open('CHECKLIST.md', 'w') as f:
        f.write('\n'.join(output_lines) + '\n')

if __name__ == '__main__':
    main()
