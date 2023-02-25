import sys
import os
from CPP14Lexer import *
from CPP14Parser import *
from CPP14ParserVisitor import *
from antlr4 import *


if __name__ == '__main__':
    #  python3.9 main.py example/hello.cpp
    input_stream = FileStream(sys.argv[1])

    # antlr -Dlanguage=Python3 CPP14Lexer.g4 
    cpplex = CPP14Lexer(input_stream)
    commtokstream = CommonTokenStream(cpplex)

    cpparser = CPP14Parser(commtokstream)
    print("parse errors: {}".format(cpparser._syntaxErrors))
