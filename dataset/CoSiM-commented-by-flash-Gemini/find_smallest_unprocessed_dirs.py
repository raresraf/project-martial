
import os

def get_dir_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

unprocessed_dirs = []
with open("all_uuid_dirs.txt", "r") as f:
    for line in f:
        dir_name = line.strip()
        checkpoint_path = os.path.join(dir_name, ".checkpoint")
        if not os.path.exists(checkpoint_path):
            size = get_dir_size(dir_name)
            unprocessed_dirs.append((dir_name, size))

unprocessed_dirs.sort(key=lambda x: x[1])

with open("smallest_unprocessed_dirs.txt", "w") as f:
    for i, (dir_name, size) in enumerate(unprocessed_dirs):
        if i >= 100:
            break
        f.write(dir_name + "
")
