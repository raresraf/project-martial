"""Package comments checks the similarity between two comments."""

# Sample run: bazel run //modules/drivers:comments_driver -- --source_files_dir=/Users/raresraf/code/project-martial/samples/comments_trivial --alsologtostderr

import datetime
from rapidfuzz import fuzz
from grammars.go.GoLexer import GoLexer, FileStream
import tempfile
import en_core_web_lg
from modules.comments_helpers import strip_comment_line_and_append_line_number
import modules.comments_helpers as comments_helpers
from sklearn.metrics.pairwise import cosine_similarity
from simple_elmo import ElmoModel
from spacy.tokens import Doc
import numpy as np


class CommentsAnalysis():
    def __init__(self):
        self.initTimestamp = datetime.datetime.now()
        self.fileDict = {}
        self.token = 'n/a'
        self.spacy_core_web = en_core_web_lg.load()

        self.enable_elmo = False
        if self.enable_elmo:
            self.elmo = ElmoModel()
            self.elmo.load("/Users/raresraf/code/project-martial/209")

    def link_to_token(self, token):
        self.token = token

    def load_file(self, filepath):
        f = open(filepath, 'r')
        source = f.read()
        self.fileDict[filepath] = source

    def load_text(self, filepath, source):
        self.fileDict[filepath] = source

    def parse(self):
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
                    strip_comment_line_and_append_line_number(
                        t.text, t.line, findings_dict[k])

            temp.close()

        # print(f"Parse of comments is complete: {findings_dict}")
        return findings_dict

    def analyze_2_files(self, display_name, similarity_method, comm_to_seq, custom_similarity=False):
        findings_dict = self.parse()
        common_list = []
        lines_in_1 = []
        lines_in_2 = []
        file1 = comm_to_seq(findings_dict["file1"])
        file2 = comm_to_seq(findings_dict["file2"])
        total_combinations = len(file1) * len(file2)
        print_report = [len(file1) / 10, 2 * len(file1) / 10, 3 * len(file1) / 10, 4 * len(file1) / 10, 5 * len(
            file1) / 10, 6 * len(file1) / 10, 7 * len(file1) / 10, 8 * len(file1) / 10, 9 * len(file1) / 10, len(file1) - 1]

        print(
            f"[traceID: {self.token}] {display_name}: need to analyze {total_combinations} sequences")
        for idx, f1 in enumerate(file1):
            for f2 in file2:
                cmp1 = f1[0]
                cmp2 = f2[0]
                if custom_similarity:
                    cmp1 = f1
                    cmp2 = f2
                is_similar, similarity = similarity_method(cmp1, cmp2)
                if is_similar:
                    # print(f'{display_name} detected similarity: {similarity}, {f1}, {f2}')
                    common_list.append((f1[0], f2[0]))
                    lines_in_1.append(f1[1])
                    lines_in_2.append(f2[1])
            if idx in print_report:
                print(
                    f"[traceID: {self.token}] {display_name}: Progress: {idx}/{len(file1)}", end='\r')
        print(f"[traceID: {self.token}] {display_name}: COMPLETED!")
        return common_list, lines_in_1, lines_in_2

    def comm_to_seq_1(self, file):
        return comments_helpers.comm_to_seq_default(file, 1)

    def exact_match_similarity(self, f1, f2):
        similarity = 0.0
        if f1 == f2:
            similarity = 1.0
        return f1 == f2, similarity

    def comm_to_seq_10(self, file):
        return comments_helpers.comm_to_seq_default(file, 10)

    def fuzzy_similarity(self, f1, f2):
        similarity = fuzz.ratio(f1, f2)
        return similarity >= 97.00, similarity

    def spacy_similarity(self, f1, f2):
        similarity = f1.similarity(f2)
        return similarity > 0.96, similarity

    def comm_to_seq_doc(self, file) -> list[tuple[Doc, int]]:
        """Similar to comm_to_seq but returns the Doc(commentary) instead of commentary: string."""
        resp = comments_helpers.comm_to_seq_default(file, 10)
        return [(self.spacy_core_web(long_comm), coming_from) for long_comm, coming_from in resp]

    def elmo_similarity(self, f1, f2):
        similarity = cosine_similarity(f1[2], f2[2])
        return similarity > 0.96, similarity

    def comm_to_seq_elmo(self, file) -> list[tuple[Doc, int]]:
        """Similar to comm_to_seq but returns the Doc(commentary) instead of commentary: string."""
        resp = comments_helpers.comm_to_seq_default(file, 1)
        ret = []
        for long_comm, coming_from in resp:
            long_comm_tensor = self.elmo.get_elmo_vector_average(long_comm, warmup=False)
            long_comm_tensor_avged = np.sum(long_comm_tensor[0][:], axis = 0)/long_comm_tensor.shape[1]
            ret.append((long_comm, coming_from,
                       long_comm_tensor_avged.reshape(1, -1)))
        return ret
