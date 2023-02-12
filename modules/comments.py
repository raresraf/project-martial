"""Package comments checks the similarity between two comments."""

# Sample run: bazel run //modules/drivers:comments_driver -- --source_files_dir=/Users/raresraf/code/project-martial/samples/comments --alsologtostderr

import datetime
import re

class CommentsAnalysis():
    def __init__(self):
        self.initTimestamp = datetime.datetime.now()
        self.fileDict = {}

    def load_file(self, filepath):
        f = open(filepath,'r')
        source = f.read()
        self.fileDict[filepath] = source

    def load_text(self, filepath, source):
        self.fileDict[filepath] = source

    def analyze(self):
        findings_dict = {}
        for k, v in self.fileDict.items():
            pattern = re.compile('(?:/\*(.*?)\*/)|(?://(.*?)\n)',re.S)
            findings = pattern.findall(v)
            findings_dict[k] = []
            for f in findings:
                findings_dict[k].append(f[1])
        return findings_dict

    def analyze_2_files(self):
        findings_dict = self.analyze()
        file1 = findings_dict["file1"]
        file2 = findings_dict["file2"]
        common_list = set(file1).intersection(file2)
        return common_list

