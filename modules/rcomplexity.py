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
        for metric in f1["metrics"].keys():
            for feature in f1["metrics"][metric]:
                if f1["metrics"][metric][feature] == f2["metrics"][metric][feature]:
                    lines_in_1.append(self.find_line_in_file(metric, feature, "file1"))
                    lines_in_2.append(self.find_line_in_file(metric, feature, "file2"))
                    
        return lines_in_1, lines_in_2
   
   
    def find_complexity_similarity(self):
        f1 = self.fileJSON["file1"]
        f2 = self.fileJSON["file2"]
        
        X = 9 * [4 * [1]]
        total_X = 36
        X_c = 9 * [4 * [1]]
        
        similarity = 0
        lines_in_1, lines_in_2 = [], []
        for metric in f1["metrics"].keys():
            a1 = 1 if f1["metrics"][metric]["FEATURE_TYPE"] == f2["metrics"][metric]["FEATURE_TYPE"] else 0
            a2 = a1 * max(0, 2 - abs(f1["metrics"][metric]["FEATURE_CONFIG"] - f2["metrics"][metric]["FEATURE_CONFIG"])) / 2
            a3 = 0
            if f1["metrics"][metric]["R-VAL"] + f2["metrics"][metric]["R-VAL"] > 0:
                a3 = a1 * a2 * (f1["metrics"][metric]["R-VAL"] + f2["metrics"][metric]["R-VAL"] - abs(f1["metrics"][metric]["R-VAL"] - f2["metrics"][metric]["R-VAL"])) / (f1["metrics"][metric]["R-VAL"] + f2["metrics"][metric]["R-VAL"])
            a4 = 0
            if f1["metrics"][metric]["INTERCEPT"] + f2["metrics"][metric]["INTERCEPT"] > 0:
                a4 = a1 * a2 * a3 * (f1["metrics"][metric]["INTERCEPT"] + f2["metrics"][metric]["INTERCEPT"] - abs(f1["metrics"][metric]["INTERCEPT"] - f2["metrics"][metric]["INTERCEPT"])) / (f1["metrics"][metric]["INTERCEPT"] + f2["metrics"][metric]["INTERCEPT"])
            
            if a1 > 1 / (36 * 3):
                lines_in_1.append(self.find_line_in_file(metric, "FEATURE_TYPE", "file1"))
                lines_in_2.append(self.find_line_in_file(metric, "FEATURE_TYPE", "file2"))
            if a2 > 1 / (36 * 4):
                lines_in_1.append(self.find_line_in_file(metric, "FEATURE_CONFIG", "file1"))
                lines_in_2.append(self.find_line_in_file(metric, "FEATURE_CONFIG", "file2"))
            if a3 > 1 / (36 * 5):
                lines_in_1.append(self.find_line_in_file(metric, "R-VAL", "file1"))
                lines_in_2.append(self.find_line_in_file(metric, "R-VAL", "file2"))
            if a4 > 1 / (36 * 6):
                lines_in_1.append(self.find_line_in_file(metric, "INTERCEPT", "file1"))
                lines_in_2.append(self.find_line_in_file(metric, "INTERCEPT", "file2"))
                
            
            X_c = X.pop() 
            similarity += X_c[0] * a1 + X_c[1] * a2 + X_c[2] * a3 + X_c[3] * a4
        similarity = similarity / total_X
        print("similarity is: ", similarity) 
        return lines_in_1, lines_in_2
        
    def find_line_in_file(self, characteristic, feature, filename):
        f_counter = 0
        while f_counter < len(self.fileDict[filename]) and characteristic not in self.fileDict[filename][f_counter]:
            f_counter += 1 
        while f_counter < len(self.fileDict[filename]) and feature not in self.fileDict[filename][f_counter]:
            f_counter += 1
        return [f_counter + 1]
