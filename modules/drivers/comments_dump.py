"""A small driver function to test comments functionality"""

from absl import app
from absl import flags
from modules.comments import CommentsAnalysis
import json
import os
import modules.comments_helpers as comments_helpers


FLAGS = flags.FLAGS
flags.DEFINE_bool("dry_run", False,
                  help="Run the project in dry-run mode.")
flags.DEFINE_integer("r", 2, help="Value of r.")
flags.DEFINE_string("source_files_dir1", "/Users/raresraf/code/examples-project-martial/kubernetes-1.2.1/pkg/api/resource",
                    help="Path to the source files1")
flags.DEFINE_string("source_files_dir2", "/Users/raresraf/code/examples-project-martial/kubernetes-1.3.1/pkg/api/resource",
                    help="Path to the source files2")
flags.DEFINE_string("source_files_dir3", "/Users/raresraf/code/examples-project-martial/kubernetes-1.25.1/pkg/api",
                    help="Path to the source files3")
flags.DEFINE_string(
    "encoding", "utf-8", help="e.g. utf-8, ISO-8859-1")
flags.DEFINE_string("extension", 'go',
                    help="The name of extension of files to join (e.g. 'go', 'cpp')")

WITH_NEXT = False

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
    for root, _, files in os.walk(FLAGS.source_files_dir3):
        for file in files:
            if not file.endswith(FLAGS.extension):
                continue
            src_full_path = os.path.join(root, file)
            file_list.append(src_full_path)

    ca = CommentsAnalysis()
    
    for f_path in file_list:
        upload_dict = {}
        with open(f_path, 'r') as f:
            upload_dict[f_path] = f.read()
            ca.load_text(f_path, upload_dict[f_path])
    
    findings_dict = ca.parse()
    comms_6 = []
    for f_path in file_list:
        for comm in comments_helpers.comm_to_seq_default(findings_dict[f_path], FLAGS.r):
            comms_6.append(comm[0])

    dmp = {}
    uid = 0
    for c in comms_6:
        uid = uid + 1
        dmp[uid] = {"comment": c, "similar_with": [uid]}
        if WITH_NEXT:
            if uid % 2 == 1:
                wuid = uid + 1
            else:
                wuid = uid - 1
            dmp[uid]["similar_with"].append(wuid)

    with open(f'/Users/raresraf/code/project-martial/dataset/comments-{FLAGS.r}-kubernetes.txt', 'w') as f:
        json.dump(dmp, f, indent=4, sort_keys=True)
    

if __name__ == '__main__':
    app.run(main)
