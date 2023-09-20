import json
import os

for type in ['comment_fuzzy_lines_files',
             'comment_spacy_core_web_lines_files',
             'comment_elmo_lines_files',
             'comment_roberta_lines_files',
             'comment_use_lines_files']:
  print(f">>{type}")
  print(10 * "xxx")
  for root, _, files in os.walk('/Users/raresraf/code/examples-project-martial/analyze/elmo-kubernetes-1.2.1-kubernetes-1.3.1'):
      for file in files:
          if not file.endswith(".go"):
              continue
          src_full_path = os.path.join(root, file)
          with open(src_full_path) as user_file:
              file_contents = user_file.read()
              parsed_json = json.loads(file_contents)
              if not parsed_json.get(type, None):
                  continue
              for entry in parsed_json[type]:
                  file1 = entry["file1_text"]
                  file2 = entry["file2_text"]
                  if file1 != file2 and (not file1 in file2) and (not file2 in file1):
                      print(file1)
                      print(file2)
                      print()
