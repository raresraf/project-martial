
import os
from pathlib import Path
from collections import defaultdict

LANGUAGE_EXTENSIONS = {
    "go": [".go"],
    "typescript": [".ts"],
    "c": [".c", ".h"],
    "java": [".java"],
    "rust": [".rs"],
    "cuda": [".cu", ".cuh"],
    "python": [".py"],
    "opencl": [".cl"],
    "cpp": [".cpp", ".hpp", ".cxx", ".hxx", ".cc", ".hh"],
}

def count_lines_in_file(file_path):
    """Counts the number of lines in a given file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for line in f)
    except Exception:
        return 0 

def main():
    raw_dir = Path("raw")
    if not raw_dir.is_dir():
        print(f"Error: Directory '{raw_dir}' not found.")
        return

    language_line_counts = defaultdict(int)
    file_counts = defaultdict(int)

    for project_dir in raw_dir.iterdir():
        if project_dir.is_dir():
            for root, _, files in os.walk(project_dir):
                for file_name in files:
                    file_path = Path(root) / file_name
                    if file_path.is_file():
                        for lang, extensions in LANGUAGE_EXTENSIONS.items():
                            if file_path.suffix.lower() in extensions:
                                line_count = count_lines_in_file(file_path)
                                language_line_counts[lang] += line_count
                                file_counts[lang] += 1
                                break

    print("--- Lines of Code by Language ---")
    total_lines = 0
    total_files = 0
    for lang, lines in language_line_counts.items():
        print(f"{lang.capitalize()}: {lines:,} lines ({file_counts[lang]} files)")
        total_lines += lines
        total_files += file_counts[lang]
    
    print(f"\nTotal: {total_lines:,} lines in {total_files:,} files")

if __name__ == "__main__":
    main()
