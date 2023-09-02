"""A small driver function to test comments functionality"""

from absl import app
from absl import flags

import modules.comments_api as comments_api
import json
import os

FLAGS = flags.FLAGS
flags.DEFINE_bool("dry_run", False,
                  help="Run the project in dry-run mode.")
flags.DEFINE_string("source_files_dir1", "/Users/raresraf/code/examples-project-martial/kubernetes-1.2.1/pkg/",
                    help="Path to the source files 1")
flags.DEFINE_string("source_files_dir2", "/Users/raresraf/code/examples-project-martial/kubernetes-1.3.1/pkg/",
                    help="Path to the source files 2")
flags.DEFINE_string("output_dir", "/Users/raresraf/code/examples-project-martial/analyze/kubernetes-1.2.1-kubernetes-1.3.1/",
                    help="Path to the source files 2")
flags.DEFINE_string(
    "encoding", "utf-8", help="e.g. utf-8, ISO-8859-1")
flags.DEFINE_string("extension", 'go',
                    help="The name of extension of files to join (e.g. 'go', 'cpp')")
flags.DEFINE_bool("ultra_fast_detection", False,
                  help="If enabled, it will only to try to match files from the same path")


def main(_):
    print("Hello to the CommentsAnalysis driver!")
    file_list1 = []
    for root, _, files in os.walk(FLAGS.source_files_dir1):
        for file in files:
            if not file.endswith(FLAGS.extension):
                continue
            src_full_path = os.path.join(root, file)
            file_list1.append(src_full_path)

    file_list2 = []
    for root, _, files in os.walk(FLAGS.source_files_dir2):
        for file in files:
            if not file.endswith(FLAGS.extension):
                continue
            src_full_path = os.path.join(root, file)
            file_list2.append(src_full_path)

    len_f1 = len(file_list1)
    len_f2 = len(file_list2)
    for id1, f1 in enumerate(file_list1):
        for id2, f2 in enumerate(file_list2):
            print(
                f"Progress: {(id1 + 1) * (id2 + 1) + id1 * len_f2} / {len_f1 * len_f2}", end='\r')
            shorter_f1 = f1.removeprefix(
                FLAGS.source_files_dir1).replace("/", "_____")
            shorter_f2 = f2.removeprefix(
                FLAGS.source_files_dir2).replace("/", "_____")
            if shorter_f1 != shorter_f2:
                continue
            upload_dict = {}
            with open(f1, 'r') as f:
                upload_dict["file1"] = f.read()
            with open(f2, 'r') as f:
                upload_dict["file2"] = f.read()
            report = comments_api.run_text(upload_dict)

            with open(f"{FLAGS.output_dir}{shorter_f1}__-__{shorter_f2}", "w") as outfile:
                json.dump(report, outfile, indent=4)


if __name__ == '__main__':
    app.run(main)
