import sys
import os
from GoLexer import *
from GoParser import *
from antlr4 import *
import pprint

pp = pprint.PrettyPrinter(indent=2)

if __name__ == '__main__':
    #  python3.9 main.py example/hello.go
    input_stream = FileStream(sys.argv[1])

    # antlr -Dlanguage=Python3 GoLexer.g4
    golex = GoLexer(input_stream)

    for t in golex.getAllTokens():
        pp.pprint(f"{t.line},{t.type}, {t.text}")

    commtokstream = CommonTokenStream(golex)

    goparser = GoParser(commtokstream)
    print("parse errors: {}".format(goparser._syntaxErrors))
