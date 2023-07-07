"""Welcome to the main function for Project Martial! (https://github.com/raresraf/project-martial)"""

from absl import app
from absl import flags
from threading import Lock
import re

from flask import Flask, request
from flask_cors import CORS
import secrets
from modules.comments import CommentsAnalysis

FLAGS = flags.FLAGS

api = Flask(__name__)
CORS(api)


@api.route("/api")
def api_root():
    return "<p>Hello, World!</p>"


custom_1 = "/Users/raresraf/code/examples-project-martial/merged/kubernetes-1.1.1.go"
custom_2 = "/Users/raresraf/code/examples-project-martial/merged/kubernetes-1.25.1.go"


def generate_token():
    token = secrets.token_hex(7)
    print(f"Received custom request with token {token}")
    return token


@api.route("/api/reserved/custom", methods=['GET'])
def custom_comments():
    ca = CommentsAnalysis()
    ca.link_to_token(generate_token())

    with open(custom_1, 'r') as f:
        upload_dict["file1"] = f.read()
        ca.load_text("file1", upload_dict["file1"])
    with open(custom_2, 'r') as f:
        upload_dict["file2"] = f.read()
        ca.load_text("file2", upload_dict["file2"])
    return comments_common(ca)


@api.route("/api/comments", methods=['GET'])
def comments():
    ca = CommentsAnalysis()
    ca.link_to_token(generate_token())

    if upload_dict.get("file1", None):
        ca.load_text("file1", upload_dict["file1"])
    if upload_dict.get("file2", None):
        ca.load_text("file2", upload_dict["file2"])
    return comments_common(ca)


def comments_common(ca):
    report = {
        "comment_exact_lines_files": [],
        "comment_fuzzy_lines_files": [],
    }

    common_list, lines_in_1, lines_in_2 = ca.analyze_2_files()
    print(
        f"[traceID: {ca.token}] common_list for analyze_2_files: found {len(common_list)} common sequences")
    for idx, _ in enumerate(common_list):
        report["comment_exact_lines_files"].append(
            {
                "file1": lines_in_1[idx],
                "file2": lines_in_2[idx],
            },
        )
        print(
            f"[traceID: {ca.token}] Finished analyzing comment_exact_lines_files: {idx}/{len(common_list)}")

    common_list, lines_in_1, lines_in_2 = ca.analyze_2_files_fuzzy()
    print(
        f"[traceID: {ca.token}] common_list for analyze_2_files_fuzzy: found {len(common_list)} common sequences")
    for idx, x in enumerate(common_list):
        report["comment_fuzzy_lines_files"].append(
            {
                "file1": lines_in_1[idx],
                "file2": lines_in_2[idx],
            },
        )
        print(
            f"[traceID: {ca.token}] Finished analyzing comment_fuzzy_lines_files: {idx}/{len(common_list)}")

    common_list, lines_in_1, lines_in_2 = ca.analyze_2_files_nlp()
    print(
        f"[traceID: {ca.token}] common_list for analyze_2_files_nlp: found {len(common_list)} common sequences")
    for idx, x in enumerate(common_list):
        report["comment_nlp_lines_files"].append(
            {
                "file1": lines_in_1[idx],
                "file2": lines_in_2[idx],
            },
        )
        print(
            f"[traceID: {ca.token}] Finished analyzing comment_nlp_lines_files: {idx}/{len(common_list)}")

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
