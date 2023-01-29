"""Package comments checks the similarity between two comments."""

# Sample run: bazel run //modules/drivers:comments_driver -- --source_files_dir=/Users/raresraf/code/project-martial/samples/comments --alsologtostderr

import datetime
import re

class CommentsAnalysis():
    def __init__(self):
        self.initTimestamp = datetime.datetime.now()
        self.fileDict = {}

    def load(self, filepath):
        f = open(filepath,'r')
        source = f.read()
        self.fileDict[filepath] = source

    def analyze(self):
        for k, v in self.fileDict.items():
            pattern = re.compile('(?:/\*(.*?)\*/)|(?://(.*?)\n)',re.S)
            findings = pattern.findall(v)
            print(findings)

