"""Welcome to the main function for Project Martial! (https://github.com/raresraf/project-martial)"""

import os
import sys

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_bool("dry_run", False,
                  help="Run the project in dry-run mode.")
flags.DEFINE_string("source_files_dir", "samples",
                    help="Path to the source files")


def main(_):
    pass


if __name__ == '__main__':
    app.run(main)
