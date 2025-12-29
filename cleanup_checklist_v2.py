import re
import os
import sys

def normalize_path(path_str):
    # path_str is like "tinyops.linalg.dot"
    # convert to "tinyops/linalg/dot.py"
    return path_str.replace('.', '/') + '.py'

def check_implementation_exists(path_str):
    # 1. Check direct file mapping
    file_path = normalize_path(path_str)
    if os.path.exists(file_path):
        if os.path.getsize(file_path) > 50:
            return True

    # 2. Check if it's a module
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

    path_match = re.search(r'', rest)
    if path_match:
        try:
            path = path_match.group(1)
        except IndexError:
            # Should not happen given the regex
            path = rest
    else:
        path = rest

    return prefix, rest, path

def main():
    try:
        with open('CHECKLIST.md', 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("CHECKLIST.md not found")
        sys.exit(1)

    unique_entries = {}
    header = []

    # Pass 1: Gather valid entries and check existence
    for line in lines:
        line = line.strip()
        if not line.startswith('- ['):
            header.append(line)
            continue

        try:
            prefix, rest, path = clean_line(line)
        except Exception as e:
            print(f"Error parsing line: {line}\n{e}")
            continue

        if path is None:
            # Malformed line or header
            header.append(line)
            continue

        desc_match = re.search(r'\[(.)\] (.*)', prefix)
        if not desc_match:
             continue

        desc = desc_match.group(2).strip()
        exists = check_implementation_exists(path)

        # Store for deduplication (key by path)
        unique_entries[path] = {
            'desc': desc,
            'rest': rest,
            'checked': exists
        }

    # Pass 2: Write back preserving order of first appearance
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
        # Reconstruct line: "- [x] desc → rest"
        # Ensure rest has backticks if it had them
        # clean_line returned 'rest' with backticks preserved but suffixes removed
        output_lines.append(f"- [{check_mark}] {entry['desc']} → {entry['rest']}")

    with open('CHECKLIST.md', 'w') as f:
        f.write('\n'.join(output_lines) + '\n')

if __name__ == '__main__':
    main()
