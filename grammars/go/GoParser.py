# Generated from GoParser.g4 by ANTLR 4.12.0
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

if __name__ is not None and "." in __name__:
    from .GoParserBase import GoParserBase
else:
    from GoParserBase import GoParserBase

def serializedATN():
    return [
        4,1,3,9,2,0,7,0,1,0,5,0,4,8,0,10,0,12,0,7,9,0,1,0,1,5,0,1,0,0,1,
        1,0,1,3,8,0,5,1,0,0,0,2,4,7,0,0,0,3,2,1,0,0,0,4,7,1,0,0,0,5,6,1,
        0,0,0,5,3,1,0,0,0,6,1,1,0,0,0,7,5,1,0,0,0,1,5
    ]

class GoParser ( GoParserBase ):

    grammarFileName = "GoParser.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [  ]

    symbolicNames = [ "<INVALID>", "COMMENT", "LINE_COMMENT", "GARBAGE" ]

    RULE_sourceFile = 0

    ruleNames =  [ "sourceFile" ]

    EOF = Token.EOF
    COMMENT=1
    LINE_COMMENT=2
    GARBAGE=3

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.12.0")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class SourceFileContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def COMMENT(self, i:int=None):
            if i is None:
                return self.getTokens(GoParser.COMMENT)
            else:
                return self.getToken(GoParser.COMMENT, i)

        def LINE_COMMENT(self, i:int=None):
            if i is None:
                return self.getTokens(GoParser.LINE_COMMENT)
            else:
                return self.getToken(GoParser.LINE_COMMENT, i)

        def GARBAGE(self, i:int=None):
            if i is None:
                return self.getTokens(GoParser.GARBAGE)
            else:
                return self.getToken(GoParser.GARBAGE, i)

        def getRuleIndex(self):
            return GoParser.RULE_sourceFile

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSourceFile" ):
                listener.enterSourceFile(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSourceFile" ):
                listener.exitSourceFile(self)




    def sourceFile(self):

        localctx = GoParser.SourceFileContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_sourceFile)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 5
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,0,self._ctx)
            while _alt!=1 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1+1:
                    self.state = 2
                    _la = self._input.LA(1)
                    if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 14) != 0)):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume() 
                self.state = 7
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,0,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





