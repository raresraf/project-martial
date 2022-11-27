"""
Script used to batch copy a set of files.

Sample usage: 
    bazel run scripts:batch_copy -- --match_file_name="4A.CPP" --walk_root="/Users/raresraf/code/TheCrawlCodeforces/results_code" --target_dir="/Users/raresraf/code/project-martial/samples/comments"
"""
import os
import sys
import shutil

from absl import app
from absl import flags


FLAGS = flags.FLAGS
flags.DEFINE_bool("dry_run", False, help="Run the batch copy in dry-run mode.")
flags.DEFINE_string(
    "target_dir", None, help="Path to the root where to store the results (e.g. /samples)")
flags.mark_flag_as_required("target_dir")
flags.DEFINE_string(
    "walk_root", None, help="Path to the root from where to search (e.g. /mydir)")
flags.mark_flag_as_required("walk_root")
flags.DEFINE_string("match_file_name", None,
                    help="The name of file to search (e.g. 4A.CPP)")
flags.mark_flag_as_required("match_file_name")


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def main(_):
    print(FLAGS.walk_root)
    for root, _, files in os.walk(FLAGS.walk_root):
        for file in files:
            if file == FLAGS.match_file_name:
                src_full_path = os.path.join(root, file)
                target_path = remove_prefix(remove_prefix(
                    src_full_path, FLAGS.walk_root), "/")
                absolute_target_path = os.path.join(
                    FLAGS.target_dir, target_path)
                if FLAGS.dry_run:
                    print(
                        f'[dry-run] COPY {src_full_path} -> {absolute_target_path}')
                    continue
                print(
                    f'MKDIR {os.path.dirname(absolute_target_path)}')
                os.makedirs(os.path.dirname(
                    absolute_target_path), exist_ok=True)
                print(f'COPY {src_full_path} -> {absolute_target_path}')
                shutil.copy(src_full_path, absolute_target_path)

if __name__ == '__main__':
    app.run(main)
