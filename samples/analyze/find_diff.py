import json
import os

for type in ['comment_fuzzy_lines_files', 'comment_spacy_core_web_lines_files']:
  print(f">>{type}")
  for root, _, files in os.walk('/Users/raresraf/code/project-martial/samples/analyze/kubernetes-1.2.1-kubernetes-1.3.1'):
    for file in files:
      if not file.endswith(".go"):
        continue
      src_full_path = os.path.join(root, file)
      # print(src_full_path)
      with open(src_full_path) as user_file:
        file_contents = user_file.read()
    
        parsed_json = json.loads(file_contents)
        for entry in parsed_json[type]:
          file1 =  entry["file1_text"]
          file2 =  entry["file2_text"]
          if file1 != file2:
            print(file1, file2)

