"""A small driver function to test comments functionality"""

from absl import app
from absl import flags

from modules.comments import CommentsAnalysis

import os

FLAGS = flags.FLAGS
flags.DEFINE_bool("dry_run", False,
                  help="Run the project in dry-run mode.")
flags.DEFINE_string("source_files_dir", "/Users/raresraf/code/project-martial/samples",
                    help="Path to the source files")


def main(_):
    print("Hello to the CommentsAnalysis driver!")
    c_a = CommentsAnalysis()

    for root, _, files in os.walk(FLAGS.source_files_dir):
        for file in files:
            src_full_path = os.path.join(root, file)
            print(f'Loading {src_full_path} in working mem')
            c_a.load_file(src_full_path)
    c_a.parse()
    
if __name__ == '__main__':
    app.run(main)
