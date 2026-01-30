
import os

def find_next_unprocessed_directory(base_path):
    all_dirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    for d in all_dirs:
        if not os.path.exists(os.path.join(base_path, d, '.checkpoint')):
            return d
    return None

base_path = '/Users/raresraf/code/project-martial/dataset/CoSiM-commented-by-flash-Gemini'
next_dir = find_next_unprocessed_directory(base_path)
if next_dir:
    print(next_dir)
else:
    print("All directories processed.")
