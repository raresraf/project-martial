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
        4,0,4,41,6,-1,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,1,0,1,0,1,0,1,0,5,
        0,14,8,0,10,0,12,0,17,9,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,5,1,26,8,1,
        10,1,12,1,29,9,1,1,2,4,2,32,8,2,11,2,12,2,33,1,2,1,2,1,3,1,3,1,3,
        1,3,1,15,0,4,1,1,3,2,5,3,7,4,1,0,2,2,0,10,10,13,13,1,0,47,47,43,
        0,1,1,0,0,0,0,3,1,0,0,0,0,5,1,0,0,0,0,7,1,0,0,0,1,9,1,0,0,0,3,21,
        1,0,0,0,5,31,1,0,0,0,7,37,1,0,0,0,9,10,5,47,0,0,10,11,5,42,0,0,11,
        15,1,0,0,0,12,14,9,0,0,0,13,12,1,0,0,0,14,17,1,0,0,0,15,16,1,0,0,
        0,15,13,1,0,0,0,16,18,1,0,0,0,17,15,1,0,0,0,18,19,5,42,0,0,19,20,
        5,47,0,0,20,2,1,0,0,0,21,22,5,47,0,0,22,23,5,47,0,0,23,27,1,0,0,
        0,24,26,8,0,0,0,25,24,1,0,0,0,26,29,1,0,0,0,27,25,1,0,0,0,27,28,
        1,0,0,0,28,4,1,0,0,0,29,27,1,0,0,0,30,32,8,1,0,0,31,30,1,0,0,0,32,
        33,1,0,0,0,33,31,1,0,0,0,33,34,1,0,0,0,34,35,1,0,0,0,35,36,6,2,0,
        0,36,6,1,0,0,0,37,38,5,47,0,0,38,39,1,0,0,0,39,40,6,3,0,0,40,8,1,
        0,0,0,4,0,15,27,33,1,0,1,0
    ]

class GoLexer(Lexer):

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    COMMENT = 1
    LINE_COMMENT = 2
    GARBAGE = 3
    GARBAGE_SLASH = 4

    channelNames = [ u"DEFAULT_TOKEN_CHANNEL", u"HIDDEN" ]

    modeNames = [ "DEFAULT_MODE" ]

    literalNames = [ "<INVALID>",
 ]

    symbolicNames = [ "<INVALID>",
            "COMMENT", "LINE_COMMENT", "GARBAGE", "GARBAGE_SLASH" ]

    ruleNames = [ "COMMENT", "LINE_COMMENT", "GARBAGE", "GARBAGE_SLASH" ]

    grammarFileName = "GoLexer.g4"

    def __init__(self, input=None, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.12.0")
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None


