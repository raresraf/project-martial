"""
Script used to batch copy a set of files.

Sample usage: 
    bazel run scripts:join_files -- --alsologtostderr --walk_root=/Users/raresraf/code/examples-project-martial/kubernetes-1.1.1 --extension=go --output_dir=/Users/raresraf/code/examples-project-martial/merged/
"""
import os

from absl import app
from absl import flags


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "walk_root", None, help="Path to the root from where to search (e.g. /mydir)")
flags.DEFINE_string(
    "output_dir", None, help="Path to the root where to output the merged file (e.g. /myoutdir)")
flags.mark_flag_as_required("walk_root")
flags.DEFINE_string("extension", None,
                    help="The name of extension of files to join (e.g. 'go', 'cpp')")
flags.mark_flag_as_required("extension")


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def main(_):
    print(FLAGS.walk_root)
    target_file = os.path.basename(os.path.normpath(FLAGS.walk_root)) + "." + FLAGS.extension
    with open(os.path.join(FLAGS.output_dir, target_file), "a") as merged_file:
        for root, _, files in os.walk(FLAGS.walk_root):
            for file in files:
                if not file.endswith(FLAGS.extension):
                    continue

                src_full_path = os.path.join(root, file)
                print(src_full_path)

                with open(src_full_path, "r") as read_file:
                    merged_file.write(f"// [MERGER]: {src_full_path}\n\n")
                    merged_file.write(read_file.read())
                merged_file.write("\n\n\n")

if __name__ == '__main__':
    app.run(main)
