"""Welcome to the main function for Project Martial! (https://github.com/raresraf/project-martial)"""

import os
import sys

from absl import app
from absl import flags

from flask import Flask, request
from flask_cors import CORS

FLAGS = flags.FLAGS

api = Flask(__name__)
CORS(api)


@api.route("/api")
def api_root():
    return "<p>Hello, World!</p>"


@api.route("/api/upload", methods=['POST'])
def upload():
    summary = {"received" : []}
    print(request.json)
    for k, v in request.json.items():
        summary["received"].append(k) 
        print(k, v)
    return summary


def main(_):
    api.run()


if __name__ == '__main__':
    app.run(main)
