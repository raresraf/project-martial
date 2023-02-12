"""Welcome to the main function for Project Martial! (https://github.com/raresraf/project-martial)"""

from absl import app
from absl import flags
from threading import Lock

from flask import Flask, request
from flask_cors import CORS

from modules.comments import CommentsAnalysis

FLAGS = flags.FLAGS

api = Flask(__name__)
CORS(api)


@api.route("/api")
def api_root():
    return "<p>Hello, World!</p>"



@api.route("/api/comments", methods=['GET'])
def comments():
    report = {"comment_exact_lines_files": []}
    ca = CommentsAnalysis()
    if upload_dict.get("file1", None):
        ca.load_text("file1", upload_dict["file1"])
    if upload_dict.get("file2", None):
        ca.load_text("file2", upload_dict["file2"])
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
        report["comment_exact_lines_files"].append({"file1": found_in_1, "file2": found_in_2})
    return report


lock_upload_dict = Lock()
upload_dict = {}


@api.route("/api/upload", methods=['POST'])
def upload():
    summary = {"received": []}
    print(request.json)
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
