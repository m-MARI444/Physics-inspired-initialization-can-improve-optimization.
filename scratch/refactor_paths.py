import os
PSSA_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def refactor_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    target = f"{PSSA_PROJECT_DIR}"
    if target not in content:
        return False

    # Determine if the file is in a subdirectory of the training workspace
    rel_path = os.path.relpath(file_path, "/home/goatrobotics/Training_AI")
    is_sub = '/' in rel_path

    # Define PSSA_PROJECT_DIR location helper
    if is_sub:
        dir_block = "PSSA_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))"
    else:
        dir_block = "PSSA_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))"

    # Insert dir_block after import os
PSSA_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if "import os
PSSA_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))" in content:
        content = content.replace("import os
PSSA_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))", "import os
PSSA_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n" + dir_block)
    else:
        content = "import os
PSSA_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n" + dir_block + "\n" + content

    # Replace target paths with dynamic f-string references
    content = content.replace('f"{PSSA_PROJECT_DIR}', 'f"{PSSA_PROJECT_DIR}')
    content = content.replace("f'{PSSA_PROJECT_DIR}", "f'{PSSA_PROJECT_DIR}")

    with open(file_path, 'w') as f:
        f.write(content)
    return True

def main():
    workspace = "/home/goatrobotics/Training_AI"
    print(f"Scanning workspace for hardcoded paths under: {workspace}")
    
    count = 0
    for root, dirs, files in os.walk(workspace):
        # Exclude build/virtualenv and cached dirs
        if '.venv' in root or '.git' in root or '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                fp = os.path.join(root, file)
                if refactor_file(fp):
                    print(f"Successfully refactored paths in: {fp}")
                    count += 1
                    
    print(f"Refactor complete. Updated {count} files.")

if __name__ == "__main__":
    main()
