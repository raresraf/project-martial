import os
import sys

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

raw_dir = 'raw'
dirs = []
for d in os.listdir(raw_dir):
    full_path = os.path.join(raw_dir, d)
    if os.path.isdir(full_path):
        if not os.path.exists(os.path.join(full_path, '.checkpoint')):
            dirs.append((get_size(full_path), d))

dirs.sort(key=lambda x: x[0])

for size, d in dirs[:50]:
    print(d)
