"""Welcome to the main function for Project Martial! (https://github.com/raresraf/project-martial)"""

from absl import app
from absl import flags
from threading import Lock
import re

from flask import Flask, request
from flask_cors import CORS

from modules.comments import CommentsAnalysis

FLAGS = flags.FLAGS

api = Flask(__name__)
CORS(api)


@api.route("/api")
def api_root():
    return "<p>Hello, World!</p>"


@api.route("/api/reserved/custom", methods=['GET'])
def custom_comments():
    ca = CommentsAnalysis()
    with open("/Users/raresraf/code/examples-project-martial/merged/kubernetes-1.1.1.go", 'r') as f:
        upload_dict["file1"] = f.read()
        ca.load_text("file1", upload_dict["file1"])
    with open("/Users/raresraf/code/examples-project-martial/merged/kubernetes-1.25.1.go", 'r') as f:
        upload_dict["file2"] = f.read()
        ca.load_text("file2", upload_dict["file2"])
    return comments_common(ca)


@api.route("/api/comments", methods=['GET'])
def comments():
    ca = CommentsAnalysis()
    if upload_dict.get("file1", None):
        ca.load_text("file1", upload_dict["file1"])
    if upload_dict.get("file2", None):
        ca.load_text("file2", upload_dict["file2"])
    return comments_common(ca)


def comments_common(ca):
    report = {"comment_exact_lines_files": [],
              "comment_fuzzy_lines_files": [],
              }
    common_list = ca.analyze_2_files()
    for x in common_list:
        found_in_1 = []
        line_count = 0
        for line in upload_dict["file1"].split("\n"):
            line_count = line_count + 1
            if x in line:
                found_in_1.append(line_count)
        line_count = 0
        found_in_2 = []
        for line in upload_dict["file2"].split("\n"):
            line_count = line_count + 1
            if x in line:
                found_in_2.append(line_count)
        report["comment_exact_lines_files"].append(
            {"file1": found_in_1, "file2": found_in_2})

    common_list = ca.analyze_2_files_fuzzy()
    for x in common_list:
        found_in_1 = []
        line_count = 0
        for line in upload_dict["file1"].split("\n"):
            line_count = line_count + 1
            if x in line:
                found_in_1.append(line_count)
        line_count = 0
        found_in_2 = []
        for line in upload_dict["file2"].split("\n"):
            line_count = line_count + 1
            if x in line:
                found_in_2.append(line_count)
        report["comment_fuzzy_lines_files"].append(
            {"file1": found_in_1, "file2": found_in_2})
    return report


lock_upload_dict = Lock()
upload_dict = {}


@api.route("/api/upload", methods=['POST'])
def upload():
    summary = {"received": []}
    for k, v in request.json.items():
        summary["received"].append(k)
        if k == "file1" or k == "file2":
            lock_upload_dict.acquire()
            upload_dict[k] = v
            lock_upload_dict.release()
    return summary


def main(_):
    api.run()


if __name__ == '__main__':
    app.run(main)
