# Generated from GoLexer.g4 by ANTLR 4.12.0
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
    from typing import TextIO
else:
    from typing.io import TextIO


def serializedATN():
    return [
        4,0,3,35,6,-1,2,0,7,0,2,1,7,1,2,2,7,2,1,0,1,0,1,0,1,0,5,0,12,8,0,
        10,0,12,0,15,9,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,5,1,24,8,1,10,1,12,
        1,27,9,1,1,2,4,2,30,8,2,11,2,12,2,31,1,2,1,2,1,13,0,3,1,1,3,2,5,
        3,1,0,2,2,0,10,10,13,13,1,0,47,47,37,0,1,1,0,0,0,0,3,1,0,0,0,0,5,
        1,0,0,0,1,7,1,0,0,0,3,19,1,0,0,0,5,29,1,0,0,0,7,8,5,47,0,0,8,9,5,
        42,0,0,9,13,1,0,0,0,10,12,9,0,0,0,11,10,1,0,0,0,12,15,1,0,0,0,13,
        14,1,0,0,0,13,11,1,0,0,0,14,16,1,0,0,0,15,13,1,0,0,0,16,17,5,42,
        0,0,17,18,5,47,0,0,18,2,1,0,0,0,19,20,5,47,0,0,20,21,5,47,0,0,21,
        25,1,0,0,0,22,24,8,0,0,0,23,22,1,0,0,0,24,27,1,0,0,0,25,23,1,0,0,
        0,25,26,1,0,0,0,26,4,1,0,0,0,27,25,1,0,0,0,28,30,8,1,0,0,29,28,1,
        0,0,0,30,31,1,0,0,0,31,29,1,0,0,0,31,32,1,0,0,0,32,33,1,0,0,0,33,
        34,6,2,0,0,34,6,1,0,0,0,4,0,13,25,31,1,0,1,0
    ]

class GoLexer(Lexer):

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    COMMENT = 1
    LINE_COMMENT = 2
    GARBAGE = 3

    channelNames = [ u"DEFAULT_TOKEN_CHANNEL", u"HIDDEN" ]

    modeNames = [ "DEFAULT_MODE" ]

    literalNames = [ "<INVALID>",
 ]

    symbolicNames = [ "<INVALID>",
            "COMMENT", "LINE_COMMENT", "GARBAGE" ]

    ruleNames = [ "COMMENT", "LINE_COMMENT", "GARBAGE" ]

    grammarFileName = "GoLexer.g4"

    def __init__(self, input=None, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.12.0")
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None


