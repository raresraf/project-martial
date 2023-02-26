"""Package comments checks the similarity between two comments."""

# Sample run: bazel run //modules/drivers:comments_driver -- --source_files_dir=/Users/raresraf/code/project-martial/samples/comments --alsologtostderr

import datetime
import re
from rapidfuzz import fuzz

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

    def analyze_2_files_fuzzy(self):
        return self.analyze_2_files_fuzzy_impl()


    def analyze_2_files_fuzzy_impl(self):
        ret = []
        findings_dict = self.analyze()
        file1 = self.comms_to_seq(findings_dict["file1"])
        file2 = self.comms_to_seq(findings_dict["file2"])
        for f1 in file1:
            for f2 in file2:
                if (fuzz.ratio(f1[0],f2[0])) > 96.66:
                    ret = ret + f1[1] + f2[1]
        print(ret)
        return ret


    """ Max 10 long sequences, consecutive."""
    def seq(self, x):
        if len(x) == 0:
            return []
        res = [(x[0],)]
        nexts = self.seq(x[1:])
        for n in nexts:
            if n[0] == x[0] + 1 and len(n) < 9:
                res = res + [(x[0],) + n]
        return res + nexts
    
    def comms_to_seq(self, file):
        l = len(file)
        resp = []
        for i in self.seq(range(l)):
            coming_from = []
            long_comm = ""
            for ii in i:
                long_comm = long_comm + re.sub(r'[^\w\s]','',file[ii])
                coming_from.append(file[ii])
            resp.append((long_comm, coming_from))
        return resp