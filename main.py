"""Welcome to the main function for Project Martial! (https://github.com/raresraf/project-martial)"""

from absl import app
from absl import flags
from flask import Flask, request
from flask_cors import CORS
from threading import Lock

import modules.comments_api as comments_api

FLAGS = flags.FLAGS

api = Flask(__name__)
CORS(api)


lock_upload_dict = Lock()
upload_dict = {}



@api.route("/api")
def api_root():
    return "<p>Hello, World!</p>"


@api.route("/api/comments", methods=['GET'])
def comments():
    return comments_api.run(upload_dict)


@api.route("/api/reserved/custom", methods=['GET'])
def custom_comments():
    custom_1 = "/Users/raresraf/code/examples-project-martial/merged/kubernetes-1.1.1.go"
    custom_2 = "/Users/raresraf/code/examples-project-martial/merged/kubernetes-1.25.1.go"
    with open(custom_1, 'r') as f:
        upload_dict["file1"] = f.read()
    with open(custom_2, 'r') as f:
        upload_dict["file2"] = f.read()
    return comments_api.run(upload_dict)




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
