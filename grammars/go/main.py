# Quick run: bazel run grammars/go:main -- --source_path="/Users/raresraf/code/project-martial/grammars/go/example/hello.go" --alsologtostderr
# Long run: bazel run grammars/go:main -- --source_path="/Users/raresraf/code/examples-project-martial/merged/kubernetes-1.1.1.go" --alsologtostderr
from GoLexer import *
from GoParser import *
from antlr4 import *
import pprint
import os
from absl import flags
from absl import app
from modules.comments_helpers import strip_comment_line
import nltk
import enchant
from nltk.corpus import wordnet
import latextable
from texttable import Texttable

FLAGS = flags.FLAGS
flags.DEFINE_string("source_path", "/Users/raresraf/code/examples-project-martial/merged/kubernetes-1.1.1.go",
                    help="The source path to gather stats for.")
flags.DEFINE_string("english_words_path", "/Users/raresraf/code/project-martial/modules/words.txt",
                    help="The source path to find words text file.")

pp = pprint.PrettyPrinter(indent=2)


def get_total_lines(path) -> int:
    count = 0
    for _ in open(path, encoding="utf8"):
        count += 1
    return count


def get_total_comments(file_path: str) -> dict:
    total_lines = get_total_lines(file_path)
    input_stream = FileStream(file_path, encoding='utf-8')
    # antlr -Dlanguage=Python3 GoLexer.g4
    lex = GoLexer(input_stream)

    count = 0
    count_line = 0
    count_multiline = 0

    start_multi_line_comment = None
    for t in lex.getAllTokens():
        if t.type != lex.COMMENT and start_multi_line_comment:
            count += t.line - start_multi_line_comment
            start_multi_line_comment = None
        if t.type == lex.COMMENT:
            count_multiline += 1
            start_multi_line_comment = t.line
        if t.type == lex.LINE_COMMENT:
            count_line += 1
            count += 1
    if start_multi_line_comment:  # file finishes in */ token.
        count += total_lines - start_multi_line_comment

    return {
        "total_lines_of_comments": count,
        "total_single_line_comments": count_line,
        "total_multi_line_comments": count_multiline,
    }


def compare_comments_to_the_rest(file_path: str) -> dict:
    input_stream = FileStream(file_path, encoding='utf-8')
    # antlr -Dlanguage=Python3 GoLexer.g4
    lex = GoLexer(input_stream)

    comments_text_chars = 0
    total_text_chars = 0
    comments_text_words = 0
    total_text_words = 0

    for t in lex.getAllTokens():
        if t.type == lex.COMMENT or t.type == lex.LINE_COMMENT:
            comments_text_chars += len(t.text)
        total_text_chars += len(t.text)
        val = t.text.split("\n")
        for nval in val:
            for sval in nval.split(" "):
                sval = strip_comment_line(sval)
                if sval == "":
                    continue
                if t.type == lex.COMMENT or t.type == lex.LINE_COMMENT:
                    comments_text_words += 1
                total_text_words += 1

    if not total_text_words:
        total_text_words = 1
    if not total_text_chars:
        total_text_chars = 1
    return {
        "number_of_comments_words": comments_text_words,
        "percentage_of_comments_words": comments_text_words / total_text_words * 100,
        "number_of_comments_chars": comments_text_chars,
        "percentage_of_comments_chars": comments_text_chars / total_text_chars * 100,
    }


def count_words_in_comments(file_path: str) -> dict:
    total_lines = get_total_lines(file_path)
    input_stream = FileStream(file_path, encoding='utf-8')
    # antlr -Dlanguage=Python3 GoLexer.g4
    lex = GoLexer(input_stream)

    with open(FLAGS.english_words_path, 'r') as file:
        word_list = [line.strip() for line in file]

    total_words = 0
    total_english_words = 0
    dictionary = enchant.Dict("en_US")
    english_words = set(nltk.corpus.words.words())
    for t in lex.getAllTokens():
        if t.type == lex.COMMENT or t.type == lex.LINE_COMMENT:
            ws = nltk.word_tokenize(t.text)
            for w in ws:
                if not w.isalpha():
                    continue
                total_words += 1
                wl = w.lower()
                if wl in word_list or dictionary.check(wl) or wordnet.synsets(wl) or wl in english_words:
                    total_english_words += 1
                #    print("In english... ", w)
                # else:
                #    print("Not in english... ", w)
        print(f"progress... {t.line} / {total_lines}")

    return {
        "number_of_comments_alpha_words": total_words,
        "number_of_english_pyenchant_words": total_english_words,
    }


def run_get_total_comments(file_path: str):
    comms_stats = get_total_comments(file_path)
    total_lines = get_total_lines(file_path)
    total_lines_of_comments = comms_stats["total_lines_of_comments"]
    total_single_line_comments = comms_stats["total_single_line_comments"]
    total_multi_line_comments = comms_stats["total_multi_line_comments"]
    percentage_of_comments = comms_stats["total_lines_of_comments"]/total_lines * 100
    
    print("===== STATS =====")
    print(f'Total lines of code: {total_lines}')
    print(f'Total lines of comments: {total_lines_of_comments}')
    print(
        f'Total number of single line comments: {total_single_line_comments}')
    print(
        f'Total number of multi-line comments: {total_multi_line_comments}')
    print(
        f'Percentage of comments line: {percentage_of_comments}')
    return total_lines, total_lines_of_comments, total_single_line_comments, total_multi_line_comments, percentage_of_comments


def run_compare_comments_to_the_rest(file_path: str):
    comms_vs_rest = compare_comments_to_the_rest(file_path)
    number_of_comments_chars = comms_vs_rest["number_of_comments_chars"]
    percentage_of_comments_chars=comms_vs_rest["percentage_of_comments_chars"]
    number_of_comments_words=comms_vs_rest["number_of_comments_words"]
    percentage_of_comments_words=comms_vs_rest["percentage_of_comments_words"]
    print(
        f'Number of comments chars: {number_of_comments_chars}')
    print(
        f'Percentage of comments chars: {percentage_of_comments_chars}')
    print(
        f'Number of comments words: {number_of_comments_words}')
    print(
        f'Percentage of comments words: {percentage_of_comments_words}')

    return number_of_comments_chars,  percentage_of_comments_chars, number_of_comments_words, percentage_of_comments_words

def run_count_words_in_comments(file_path: str):
    comms_words = count_words_in_comments(file_path)
    number_of_comments_alpha_words = comms_words["number_of_comments_alpha_words"]
    number_of_english_pyenchant_words = comms_words["number_of_english_pyenchant_words"]
    percentage_of_alpha_comments_words_valid_english = comms_words["number_of_english_pyenchant_words"] / comms_words["number_of_comments_alpha_words"] * 100

    print(
        f'Number of alpha comments words: {number_of_comments_alpha_words}')
    print(
        f'Number of english pyenchant alpha comments words: {number_of_english_pyenchant_words}')
    print(
        f'Percentage of alpha comments words that are valid english words: {percentage_of_alpha_comments_words_valid_english}')

    return number_of_comments_alpha_words, number_of_english_pyenchant_words, percentage_of_alpha_comments_words_valid_english

def run_stats(file_path: str):
    total_lines, total_lines_of_comments, total_single_line_comments, total_multi_line_comments, percentage_of_comments = run_get_total_comments(file_path)
    number_of_comments_chars,  percentage_of_comments_chars, number_of_comments_words, percentage_of_comments_words = run_compare_comments_to_the_rest(file_path)
    number_of_comments_alpha_words, number_of_english_pyenchant_words, percentage_of_alpha_comments_words_valid_english = run_count_words_in_comments(file_path)
    return total_lines, total_lines_of_comments, total_single_line_comments, total_multi_line_comments, percentage_of_comments, number_of_comments_chars,  percentage_of_comments_chars, number_of_comments_words, percentage_of_comments_words, number_of_comments_alpha_words, number_of_english_pyenchant_words, percentage_of_alpha_comments_words_valid_english

def main(_):
    file_path = FLAGS.source_path
    _, project = os.path.split(file_path)
    total_lines, total_lines_of_comments, total_single_line_comments, total_multi_line_comments, percentage_of_comments, number_of_comments_chars,  percentage_of_comments_chars, number_of_comments_words, percentage_of_comments_words, number_of_comments_alpha_words, number_of_english_pyenchant_words, percentage_of_alpha_comments_words_valid_english = run_stats(file_path)

    # We use only lexer data.
    # commtokstream = CommonTokenStream(golex)
    # goparser = GoParser(commtokstream)
    # print("parse errors: {}".format(goparser._syntaxErrors))

    latex_table(project, total_lines, total_lines_of_comments, total_single_line_comments, total_multi_line_comments, percentage_of_comments, number_of_comments_chars,  percentage_of_comments_chars, number_of_comments_words, percentage_of_comments_words, number_of_comments_alpha_words, number_of_english_pyenchant_words, percentage_of_alpha_comments_words_valid_english)

def latex_table(project, total_lines, total_lines_of_comments, total_single_line_comments, total_multi_line_comments, percentage_of_comments, number_of_comments_chars,  percentage_of_comments_chars, number_of_comments_words, percentage_of_comments_words, number_of_comments_alpha_words, number_of_english_pyenchant_words, percentage_of_alpha_comments_words_valid_english):
    table_1 = Texttable()
    table_1.set_cols_align(6 * ["X"])
    table_1.set_cols_valign(6 * ["t"])
    table_1.add_rows([["Project",
 "Total lines of comments/code", 
 "Total number of single / multi line comments",
 "Number of comments chars",
 "Number of comments words",
 "Number of english words / total words in alpha comments",
 ],
                     [project,
                      f"{total_lines_of_comments}/{total_lines} ({percentage_of_comments:.2f}\%)", 
                      f"{total_single_line_comments}/{total_multi_line_comments}", 
                      f"{number_of_comments_chars} ({percentage_of_comments_chars:.2f}\%)",
                      f"{number_of_comments_words} ({percentage_of_comments_words:.2f}\%)",
                      f"{number_of_english_pyenchant_words}/{number_of_comments_alpha_words} ({percentage_of_alpha_comments_words_valid_english:.2f}\%)",
                      ],
                    ])
    print('Texttable Output:')
    print(table_1.draw())
    print('\nLatextable Output:')
    print(latextable.draw_latex(table_1, caption="An example table.", label="table:example_table"))

if __name__ == '__main__':
    app.run(main)
