"""Package comments checks the similarity between two comments."""

# Sample run: bazel run //modules/drivers:comments_driver -- --source_files_dir=/Users/raresraf/code/project-martial/samples/comments_trivial --alsologtostderr

import datetime
import re
from rapidfuzz import fuzz
from grammars.go.GoLexer import GoLexer, FileStream
import tempfile
import spacy
import en_core_web_sm

class CommentsAnalysis():
    def __init__(self):
        self.initTimestamp = datetime.datetime.now()
        self.fileDict = {}
        self.token = 'n/a'
        self.nlp = en_core_web_sm.load()

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

            input_stream = FileStream(temp.name, encoding='utf-8')
            lex = GoLexer(input_stream)

            for t in lex.getAllTokens():
                if t.type == lex.COMMENT or t.type == lex.LINE_COMMENT:
                    strip_comment_line_and_append_line_number(t.text, t.line, findings_dict[k])

            temp.close()

        print(f"Parse of comments is complete: {findings_dict}")
        return findings_dict

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

    def comms_to_seq(self, file):
        l = len(file)
        resp = []
        for i in seq(range(l)):
            coming_from = []
            long_comm = ""
            for ii in i:
                long_comm = long_comm + re.sub(r'[^\w\s]', '', file[ii][0])
                coming_from.append(file[ii][1])
            resp.append((long_comm, coming_from))
        return resp

    def comms_to_seq_doc(self, file):
        resp = self.comms_to_seq(file)
        return [(self.nlp(long_comm), coming_from) for long_comm, coming_from in resp]

    def analyze_2_files_nlp(self):
        return self.analyze_2_files_nlp_impl()

    def analyze_2_files_nlp_impl(self):
        ret = []
        lines_in_1 = []
        lines_in_2 = []
        findings_dict = self.analyze()
        file1 = self.comms_to_seq_doc(findings_dict["file1"])
        file2 = self.comms_to_seq_doc(findings_dict["file2"])
        print(
            f"[traceID: {self.token}] analyze_2_files_nlp_impl: need to analyze {len(file1)} x {len(file2)} sequences")
        for f1 in file1:
            for f2 in file2:
                similarity = f1[0].similarity(f2[0])
                print(f"f1[0] = {f1[0]}, f2[0] = {f2[0]}, similarity={similarity}")
                if similarity > 0.9:
                    lines_in_1.append(f1[1])
                    lines_in_2.append(f2[1])
                    ret.append((f1[0], f2[0]))


        # Try Word2Vec + spaCy
        return ret, lines_in_1, lines_in_2


def strip_comment_line_and_append_line_number(val, line, to_append):
    val = val.split("\n")
    for sval in val:
        sval = strip_comment_line(sval)
        if sval == "":
            continue
        to_append.append((sval, line))
        line = line + 1

def strip_comment_line(val):
    val = val.strip("\t").strip("\n").lstrip("/* ").rstrip("/* ")
    if val == '':
        return ""
    if val == '//':
        return ""
    if val == '/*':
        return ""
    if val == '*/':
        return ""
    return val

def seq(x):
    """ Max 10 long sequences, consecutive."""
    if len(x) == 0:
        return []
    res = [(x[0],)]
    nexts = seq(x[1:])
    for n in nexts:
        if n[0] == x[0] + 1 and len(n) < 9:
            res = res + [(x[0],) + n]
    return res + nexts
