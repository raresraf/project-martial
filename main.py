"""Welcome to the main function for Project Martial! (https://github.com/raresraf/project-martial)"""

from absl import app
from absl import flags
from flask import Flask, request
from flask_cors import CORS
from threading import Lock

import modules.comments_config as comments_config
import modules.comments_api as comments_api
import modules.comments as comments


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


@api.route("/api/comments/flags", methods=['POST'])
def set_comments_flags():
    for k, v in request.json.items():
        if not type(k) == str:
            continue
        if "enable_" in k.lower():
            if "word2vec" in k.lower():
                comments_config.config.set_enable_word2vec(v)
                print(f"ENABLE_WORD2VEC set to {comments_config.config.enable_word2vec()}")
            if "elmo" in k.lower():
                comments_config.config.set_enable_elmo(v)
                print(f"ENABLE_ELMO set to {comments_config.config.enable_elmo()}")
            if "roberta" in k.lower():
                comments_config.config.set_enable_roberta(v)
                print(f"ENABLE_ROBERTA set to {comments_config.config.enable_roberta()}")
            if "use" in k.lower():
                comments_config.config.set_enable_use(v)
                print(f"ENABLE_USE set to {comments_config.config.enable_use()}")
        if "threshold_" in k.lower():
            if "word2vec" in k.lower():
                comments_config.config.set_threshold_word2vec(v)
                print(f"THRESHOLD_WORD2VEC set to {comments_config.config.threshold_word2vec()}")
            if "elmo" in k.lower():
                comments_config.config.set_threshold_elmo(v)
                print(f"THRESHOLD_ELMO set to {comments_config.config.threshold_elmo()}")
            if "roberta" in k.lower():
                comments_config.config.set_threshold_roberta(v)
                print(f"THRESHOLD_ROBERTA set to {comments_config.config.threshold_roberta()}")
            if "use" in k.lower():
                comments_config.config.set_threshold_use(v)
                print(f"THRESHOLD_USE set to {comments_config.config.threshold_use()}")
    return {"status": "OK"}


@api.route("/api/reserved/custom", methods=['GET'])
def custom_comments():
    custom_1 = "/Users/raresraf/code/examples-project-martial/merged/baby-kubernetes-1.2.1.go"
    custom_2 = "/Users/raresraf/code/examples-project-martial/merged/baby-kubernetes-1.3.1.go"
    lock_upload_dict.acquire()
    with open(custom_1, 'r') as f:
        upload_dict["file1"] = f.read()
    with open(custom_2, 'r') as f:
        upload_dict["file2"] = f.read()
    lock_upload_dict.release()
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
