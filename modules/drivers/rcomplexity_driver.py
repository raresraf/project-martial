import os
import json
from absl import app
import modules.rcomplexity as rcomplexity

dataset = {}
outcome = []

def read_all_json_files_recursive(root_path):
    for root, _, files in os.walk(root_path):
        for file in files:
            if file.endswith('PROCESSED.RAF'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    problem = os.path.basename(os.path.dirname(file_path))
                    if "_" in problem:
                        continue
                    if not dataset.get(problem, None):
                        dataset[problem] = []
                    dataset[problem].append(data)

def main(_):
    root_directory_path = '/Users/raresraf/code/TheOutputsCodeforces/processed/atomic_perf/'
    read_all_json_files_recursive(root_directory_path)

    for k in dataset.keys():
        print(k, len(dataset[k]))

    for k1 in dataset.keys():
        for k2 in dataset.keys():
            for f1 in dataset[k1]:
                if not f1.get("path", None):
                    continue
                for f2 in dataset[k2]:
                    if not f2.get("path", None):
                        continue
                    print(f'Comparing: {f1["path"]} v. {f2["path"]}')
                    rca = rcomplexity.RComplexityAnalysis()
                    rca.disable_find_line = True
                    rca.fileJSON["file1"] = f1
                    rca.fileJSON["file2"] = f2
                    _, _, similarity = rca.find_complexity_similarity()
                    o = (similarity, k1, k2, f1["path"], f2["path"])
                    print(o)
                    outcome.append(o)
                    #if(len(outcome)) > 1000:
                    #    with open("/Users/raresraf/code/project-martial/tmp_rcomplexity_dataset_results.txt", 'w') as file:
                    #        for item in outcome:
                    #            file.write(f"{item}\n")
                    #        return
                        

    with open("/Users/raresraf/code/project-martial/rcomplexity_dataset_results.txt", 'w') as file:
        for item in outcome:
            file.write(f"{item}\n")
        
if __name__ == '__main__':
    app.run(main)
