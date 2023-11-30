import datetime
import json

class RComplexityAnalysis():
    def __init__(self):
        self.initTimestamp = datetime.datetime.now()
        self.fileDict = {}
        self.fileJSON = {}
        self.token = 'n/a'

    def link_to_token(self, token):
        self.token = token

    def load_birthmark(self, filepath, source):
        self.fileDict[filepath] = source.splitlines()
        self.fileJSON[filepath] = json.loads(source)
        
    def find_critical_matches(self):
        f1 = self.fileJSON["file1"]
        f2 = self.fileJSON["file2"]
        
        lines_in_1, lines_in_2 = [], []
        for k in f1["metrics"].keys():
            for feature in f1["metrics"][k]:
                if f1["metrics"][k][feature] == f2["metrics"][k][feature]:
                    lines_in_1.append(self.find_line_in_file(k, feature, "file1"))
                    lines_in_2.append(self.find_line_in_file(k, feature, "file2"))
                    
        return lines_in_1, lines_in_2
                    
    def find_line_in_file(self, characteristic, feature, filename):
        f_counter = 0
        while f_counter < len(self.fileDict[filename]) and characteristic not in self.fileDict[filename][f_counter]:
            f_counter += 1 
        while f_counter < len(self.fileDict[filename]) and feature not in self.fileDict[filename][f_counter]:
            f_counter += 1
        return [f_counter + 1]
