
import os
import sys
import re

# Assume find_next_dir.py is in the same directory and can be imported
from find_next_dir import find_next_unprocessed_directory

def get_language_from_extension(filename):
    """
    Determines the programming language based on file extension.
    """
    _, ext = os.path.splitext(filename)
    if ext in ['.c', '.h']:
        return 'c'
    elif ext in ['.cpp', '.hpp', '.cxx', '.hxx', '.cc', '.hh']:
        return 'cpp'
    elif ext == '.py':
        return 'python'
    elif ext == '.go':
        return 'go'
    elif ext in ['.ts', '.tsx']:
        return 'typescript'
    elif ext == '.java':
        return 'java'
    elif ext == '.rs':
        return 'rust'
    elif ext == '.cl':
        return 'opencl'
    return 'unknown'

def add_comments_to_code(file_path, language):
    print(f"Processing {file_path} as {language}...")
    with open(file_path, 'r') as f:
        content_lines = f.readlines()

    modified_content = []
    has_file_header = False

    # Check for existing file-level header comment
    if language in ['c', 'cpp', 'java', 'typescript']:
        for i, line in enumerate(content_lines):
            stripped_line = line.strip()
            if stripped_line.startswith("/*") or stripped_line.startswith("/**") or stripped_line.startswith("//"):
                if i == 0 or (i > 0 and content_lines[i-1].strip() == ""):
                    has_file_header = True
                    break
            elif stripped_line:
                break
    elif language == 'python':
        for i, line in enumerate(content_lines):
            stripped_line = line.strip()
            if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                has_file_header = True
                break
            elif stripped_line:
                break
    elif language == 'rust':
        for i, line in enumerate(content_lines):
            stripped_line = line.strip()
            if stripped_line.startswith('//!'):
                has_file_header = True
                break
            elif stripped_line:
                break
    elif language == 'go':
        for i, line in enumerate(content_lines):
            stripped_line = line.strip()
            if stripped_line.startswith('//') or stripped_line.startswith('/*'):
                has_file_header = True
                break
            elif stripped_line:
                break

    # Add file-level header if missing
    if not has_file_header:
        file_name = os.path.basename(file_path)
        if language == 'c' or language == 'cpp' or language == 'opencl':
            header = [
                f"/**\n",
                f" * @file {file_name}\n",
                f" * @brief Semantic documentation for {file_name}.\n",
                " *        This is a placeholder. Detailed semantic analysis will be applied later.\n",
                " */\n"
            ]
        elif language == 'python':
            header = [
                f'"""\n',
                f'Module: {file_name}\n',
                f'Description: Semantic documentation for {file_name}.\n',
                f'             Detailed semantic analysis will be applied later.\n',
                f'"""\n'
            ]
        elif language == 'java':
            header = [
                f"/**\n",
                f" * @file {file_name}\n",
                f" * @brief Semantic documentation for {file_name}.\n",
                " *        Detailed semantic analysis will be applied later.\n",
                " */\n"
            ]
        elif language == 'go':
            header = [
                f"// {file_name}\n",
                f"// Semantic documentation for {file_name}.\n",
                f"// Detailed semantic analysis will be applied later.\n"
            ]
        elif language == 'rust':
            header = [
                f"//! {file_name}\n",
                f"//! Semantic documentation for {file_name}.\n",
                f"//! Detailed semantic analysis will be applied later.\n"
            ]
        elif language == 'typescript':
            header = [
                f"/**\n",
                f" * @file {file_name}\n",
                f" * @brief Semantic documentation for {file_name}.\n",
                " *        Detailed semantic analysis will be applied later.\n",
                " */\n"
            ]
        else:
            header = []

        if header:
            modified_content.extend(header)

    modified_content.extend(content_lines)

    block_patterns = {
        'for': r'^\s*(for\s*\(.*\)|for\s+.*in\s+.*:|for\s+await\s*\(.*\))',
        'while': r'^\s*while\s*\(?.*:?',
        'if': r'^\s*(if\s*\(.*\)|if\s+.*:)',
        'else if': r'^\s*(else\s+if\s*\(.*\)|elif\s+.*:)',
        'else': r'^\s*(else|else:)',
        'switch': r'^\s*switch\s*\(.*\)',
        'func_def': r'^\s*(?:(?:public|private|protected|static|async)\s+)?(?:[\w<>,\[\]]+\s+)?\s*(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,]+)?\s*{',
        'python_def': r'^\s*def\s+\w+\s*\(.*?\)\s*:',
        'go_func': r'^\s*func\s+(?:\(.*?\)\s*)?\w+\s*\(.*?\)\s*(?:[\w\s*]+)?\s*{',
        'rust_fn': r'^\s*fn\s+\w+\s*\(.*?\)\s*->\s*.*?\s*{',
        'ts_func': r'^\s*(?:(?:public|private|protected|static|async)\s+)?(?:[\w<>,\[\]]+\s*:\s*)?(\w+)\s*\([^)]*\)\s*:\s*[\w<>,\[\]]+\s*{'
    }

    def add_block_comment(lines, lang):
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped_line = line.strip()
            
            if stripped_line.startswith('//') or stripped_line.startswith('/*') or stripped_line.startswith('"""') or stripped_line.startswith("'''") or stripped_line.startswith('//!') or stripped_line.startswith('#'):
                new_lines.append(line)
                i += 1
                continue

            comment_to_add = None
            indent = re.match(r'^\s*', line).group(0)

            if language == 'java' and re.match(block_patterns['func_def'], line):
                if not any(lines[j].strip().startswith('/**') for j in range(max(0, i-3), i)):
                    func_name_match = re.search(r'\s*(\w+)\s*\([^)]*\)\s*{', line)
                    if func_name_match:
                        func_name = func_name_match.group(1)
                        comment_to_add = f"{indent}/**\n{indent} * @brief [Functional Utility for {func_name}]: Describe purpose here.\n{indent} */\n"
            elif language == 'c' and re.match(block_patterns['func_def'], line):
                 if not any(lines[j].strip().startswith('/**') for j in range(max(0, i-3), i)):
                    func_name_match = re.search(r'\s*(\w+)\s*\([^)]*\)\s*{', line)
                    if func_name_match:
                        func_name = func_name_match.group(1)
                        comment_to_add = f"{indent}/**\n{indent} * @brief [Functional Utility for {func_name}]: Describe purpose here.\n{indent} */\n"
            elif language == 'cpp' and re.match(block_patterns['func_def'], line):
                 if not any(lines[j].strip().startswith('/**') for j in range(max(0, i-3), i)):
                    func_name_match = re.search(r'\s*(\w+)\s*\([^)]*\)\s*{', line)
                    if func_name_match:
                        func_name = func_name_match.group(1)
                        comment_to_add = f"{indent}/**\n{indent} * @brief [Functional Utility for {func_name}]: Describe purpose here.\n{indent} */\n"
            elif language == 'python' and re.match(block_patterns['python_def'], line):
                 if not any(lines[j].strip().startswith('"""') or lines[j].strip().startswith("'''") for j in range(max(0, i-3), i)):
                    func_name_match = re.search(r'def\s+(\w+)\s*\(.*?\):', line)
                    if func_name_match:
                        func_name = func_name_match.group(1)
                        comment_to_add = f"{indent}'''\n{indent}Functional Utility: Describe purpose of {func_name} here.\n{indent}'''\n"
            elif language == 'go' and re.match(block_patterns['go_func'], line):
                 if not any(lines[j].strip().startswith('//') for j in range(max(0, i-3), i)):
                    func_name_match = re.search(r'func\s+(?:\(.*?\)\s*)?(\w+)\s*\(.*?\)\s*(?:[\w\s*]+)?\s*{', line)
                    if func_name_match:
                        func_name = func_name_match.group(1)
                        comment_to_add = f"{indent}// Functional Utility: Describe purpose of {func_name} here.\n"
            elif language == 'rust' and re.match(block_patterns['rust_fn'], line):
                 if not any(lines[j].strip().startswith('///') or lines[j].strip().startswith('/**') for j in range(max(0, i-3), i)):
                    func_name_match = re.search(r'fn\s+(\w+)\s*\(.*?\)\s*->\s*.*?\s*{', line)
                    if func_name_match:
                        func_name = func_name_match.group(1)
                        comment_to_add = f"{indent}/// Functional Utility: Describe purpose of {func_name} here.\n"
            elif language == 'typescript' and re.match(block_patterns['ts_func'], line):
                 if not any(lines[j].strip().startswith('/**') for j in range(max(0, i-3), i)):
                    func_name_match = re.search(r'function\s+(\w+)\s*\(.*?\)\s*:', line)
                    if func_name_match:
                        func_name = func_name_match.group(1)
                        comment_to_add = f"{indent}/**\n{indent} * Functional Utility: Describe purpose of {func_name} here.\n{indent} */\n"
            
            elif re.match(block_patterns['for'], line) or \
                 re.match(block_patterns['while'], line) or \
                 re.match(block_patterns['if'], line) or \
                 re.match(block_patterns['else if'], line) or \
                 re.match(block_patterns['else'], line) or \
                 re.match(block_patterns['switch'], line):
                
                if not any(lines[j].strip().startswith('//') or lines[j].strip().startswith('/*') or lines[j].strip().startswith('#') or lines[j].strip().startswith('"""') or lines[j].strip().startswith("'''") for j in range(max(0, i-1), i)):
                    if language in ['c', 'cpp', 'java', 'typescript', 'opencl']:
                        comment_to_add = f"{indent}// Block Logic: Describe purpose of this block, e.g., iteration, conditional execution\n{indent}// Invariant: State condition that holds true before and after each iteration/execution\n"
                    elif language == 'python':
                        comment_to_add = f"{indent}# Block Logic: Describe purpose of this block, e.g., iteration, conditional execution\n{indent}# Invariant: State condition that holds true before and after each iteration/execution\n"
                    elif language == 'go':
                        comment_to_add = f"{indent}// Block Logic: Describe purpose of this block, e.g., iteration, conditional execution\n{indent}// Invariant: State condition that holds true before and after each iteration/execution\n"
                    elif language == 'rust':
                        comment_to_add = f"{indent}/// Block Logic: Describe purpose of this block, e.g., iteration, conditional execution\n{indent}/// Invariant: State condition that holds true before and after each iteration/execution\n"
            
            if comment_to_add:
                new_lines.append(comment_to_add)
            new_lines.append(line)
            i += 1
        return new_lines

    content_with_block_comments = add_block_comment(modified_content, language)

    with open(file_path, 'w') as f:
        f.writelines(content_with_block_comments)
    print(f"Finished processing {file_path}")

def process_directories(base_path, max_dirs=100):
    """
    Iterates through directories, processes code files, and creates checkpoint files.
    """
    processed_count = 0
    while processed_count < max_dirs:
        next_dir_name = find_next_unprocessed_directory(base_path)
        if not next_dir_name:
            print("All available directories have been processed or checkpointed.")
            break

        current_dir_path = os.path.join(base_path, next_dir_name)
        print(f"--- Processing directory: {current_dir_path} ({processed_count + 1}/{max_dirs}) ---")

        for root, _, files in os.walk(current_dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                language = get_language_from_extension(file)
                
                if language != 'unknown' and file != '.checkpoint':
                    add_comments_to_code(file_path, language)

        checkpoint_path = os.path.join(current_dir_path, '.checkpoint')
        with open(checkpoint_path, 'w') as f:
            f.write(f"Processed on {os.path.getmtime(current_dir_path)}")
        print(f"Created checkpoint for {current_dir_path}")
        processed_count += 1

if __name__ == "__main__":
    base_path = os.getcwd()
    process_directories(base_path)
