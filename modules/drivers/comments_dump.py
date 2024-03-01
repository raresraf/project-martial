"""A small driver function to test comments functionality"""

from absl import app
from absl import flags
from modules.comments import CommentsAnalysis
import json
import os


FLAGS = flags.FLAGS
flags.DEFINE_bool("dry_run", False,
                  help="Run the project in dry-run mode.")
flags.DEFINE_string("source_files_dir1", "/Users/raresraf/code/examples-project-martial/kubernetes-1.2.1/pkg/api/resource",
                    help="Path to the source files1")
flags.DEFINE_string("source_files_dir2", "/Users/raresraf/code/examples-project-martial/kubernetes-1.3.1/pkg/api/resource",
                    help="Path to the source files2")
flags.DEFINE_string(
    "encoding", "utf-8", help="e.g. utf-8, ISO-8859-1")
flags.DEFINE_string("extension", 'go',
                    help="The name of extension of files to join (e.g. 'go', 'cpp')")


def main(_):
    print("Hello to the Comments Dump driver!")
    
    file_list = []
    for root, _, files in os.walk(FLAGS.source_files_dir1):
        for file in files:
            if not file.endswith(FLAGS.extension):
                continue
            src_full_path = os.path.join(root, file)
            file_list.append(src_full_path)
    for root, _, files in os.walk(FLAGS.source_files_dir2):
        for file in files:
            if not file.endswith(FLAGS.extension):
                continue
            src_full_path = os.path.join(root, file)
            file_list.append(src_full_path)

    ca = CommentsAnalysis()
    
    # total_len = len(file_list)
    # i = 0
    for f_path in file_list:
        upload_dict = {}
        with open(f_path, 'r') as f:
            # i = i + 1
            # print(f"Progress: {i}/{total_len}... f_path={f_path}")
            upload_dict[f_path] = f.read()
            ca.load_text(f_path, upload_dict[f_path])
    
    findings_dict = ca.parse()
    comms_6 = []
    for f_path in file_list:
        for comm in ca.comm_to_seq_2(findings_dict[f_path]):
            comms_6.append(comm[0])

    dmp = {}
    uid = 0
    for c in comms_6:
        uid = uid + 1
        dmp[uid] = {"comment": c, "similar_with": [uid]}

    with open('/Users/raresraf/code/project-martial/dataset/comments-6-kubernetes.txt', 'w') as f:
        json.dump(dmp, f, indent=4, sort_keys=True)
    

if __name__ == '__main__':
    app.run(main)
