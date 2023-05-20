"""Package comments checks the similarity between two comments."""

# Sample run: bazel run //modules/drivers:comments_driver -- --source_files_dir=/Users/raresraf/code/project-martial/samples/comments --alsologtostderr

import datetime
import re
from rapidfuzz import fuzz
from io import StringIO
from grammars.go.GoLexer import *
import tempfile

class CommentsAnalysis():
    def __init__(self):
        self.initTimestamp = datetime.datetime.now()
        self.fileDict = {}
        self.token = 'n/a'

    def link_to_token(self, token):
        self.token = token

    def load_file(self, filepath):
        f = open(filepath, 'r')
        source = f.read()
        self.fileDict[filepath] = source

    def load_text(self, filepath, source):
        self.fileDict[filepath] = source

    def analyze(self):
        findings_dict = {}
        for k, v in self.fileDict.items():
            findings_dict[k] = []

            temp = tempfile.NamedTemporaryFile()
            temp.write(bytes(v, 'utf-8'))
            temp.seek(0)

            input_stream = FileStream(temp.name)
            lex = GoLexer(input_stream)

            for t in lex.getAllTokens():
                if t.type == lex.COMMENT or t.type == lex.LINE_COMMENT:
                    self.throw_bogus(t.text	, t.line, findings_dict[k])

            temp.close()

        print(f"Parse of comments is complete: {findings_dict}")
        return findings_dict

    def throw_bogus(self, val, line, to_append):
        val = val.split("\n")
        for sval in val:
            sval = sval.strip("\t").strip("\n").lstrip("/* ").rstrip("/* ")
            if sval == '':
                continue
            if sval == '//':
                continue
            if sval == '/*':
                continue
            if sval == '*/':
                continue
            to_append.append((sval, line))
            line = line + 1

    def analyze_2_files(self):
        findings_dict = self.analyze()
        file1 = findings_dict["file1"]
        file2 = findings_dict["file2"]
        common_list = []
        lines_in_1 = []
        lines_in_2 = []
        for f1 in file1:
            for f2 in file2:
                if f1[0] == f2[0]:
                    common_list.append(f1[0])
                    lines_in_1.append((f1[1],))
                    lines_in_2.append((f2[1],)) 
        print(f"[traceID: {self.token}] Intersection finished!")
        return common_list, lines_in_1, lines_in_2

    def analyze_2_files_fuzzy(self):
        return self.analyze_2_files_fuzzy_impl()

    def analyze_2_files_fuzzy_impl(self):
        ret = []
        findings_dict = self.analyze()
        lines_in_1 = []
        lines_in_2 = []
        file1 = self.comms_to_seq(findings_dict["file1"])
        file2 = self.comms_to_seq(findings_dict["file2"])
        print(
            f"[traceID: {self.token}] analyze_2_files_fuzzy_impl: need to analyze {len(file1)} x {len(file2)} sequences")
        for f1 in file1:
            for f2 in file2:
                if (fuzz.ratio(f1[0], f2[0])) > 96.66:
                    ret.append((f1[0], f2[0]))
                    lines_in_1.append(f1[1])
                    lines_in_2.append(f2[1])
                    
        print(f"fuzzy detected: {ret}")
        return ret, lines_in_1, lines_in_2

    def seq(self, x):
        """ Max 10 long sequences, consecutive."""
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
                long_comm = long_comm + re.sub(r'[^\w\s]', '', file[ii][0])
                coming_from.append(file[ii][1])
            resp.append((long_comm, coming_from))
        return resp

