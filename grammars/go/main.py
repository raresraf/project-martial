import sys
import os
from GoLexer import *
from GoParser import *
from antlr4 import *
import pprint

pp = pprint.PrettyPrinter(indent=2)


def get_total_lines(path) -> int:
    count = 0
    for _ in open(path, encoding="utf8"):
        count += 1
    return count


def get_total_comments(lex: GoLexer, total_lines: int) -> dict:
    count = 0
    count_line = 0
    count_multiline = 0

    start_multi_line_comment = None
    for t in lex.getAllTokens():
        if t.type != lex.COMMENT and start_multi_line_comment:
            count = count + t.line - start_multi_line_comment
            start_multi_line_comment = None
        if t.type == lex.COMMENT:
            count_multiline = count_multiline + 1
            start_multi_line_comment = t.line
        if t.type == lex.LINE_COMMENT:
            count_line = count_line + 1
            count = count + 1
    if start_multi_line_comment:  # file finishes in */ token.
        count = count + total_lines - start_multi_line_comment

    return {
        "total_lines_of_comments": count,
        "total_single_line_comments": count_line,
        "total_multi_line_comments": count_multiline,
    }


if __name__ == '__main__':
    #  python3.9 main.py example/hello.go
    if len(sys.argv) != 2:
        print("Please re-run with appropriate arguments. e.g. python3.9 main.py example/hello.go")
        sys.exit(1)

    total_lines = get_total_lines(sys.argv[1])
    input_stream = FileStream(sys.argv[1], encoding='utf-8')

    # antlr -Dlanguage=Python3 GoLexer.g4
    golex = GoLexer(input_stream)

    # for t in golex.getAllTokens():
    #     pp.pprint(f"{t.line}, {t.type}, {t.text}")
    comms = get_total_comments(golex, total_lines)

    print("===== STATS =====")
    print(f'Total lines of code: {total_lines}')
    print(f'Total lines of comments: {comms["total_lines_of_comments"]}')
    print(
        f'Total number of single line comments: {comms["total_single_line_comments"]}')
    print(
        f'Total number of multi-line comments: {comms["total_multi_line_comments"]}')

    # We use only lexer data.
    # commtokstream = CommonTokenStream(golex)
    # goparser = GoParser(commtokstream)
    # print("parse errors: {}".format(goparser._syntaxErrors))
