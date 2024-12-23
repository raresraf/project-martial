import os
import re
import pprint
import matplotlib.pyplot as plt
import numpy as np

def generate_tables(data):
  """
  Generates a LaTeX table with all combinations of database engines and their similarity scores.

  Args:
    data: A list of dictionaries, where each dictionary contains information 
          about a pair of database engines and their similarity scores.
  """
  def sort_key(engine):
    return engine.replace('9.', '09.') # PG 9.6 hack
  
  engines = sorted(({d['database engine 1'] for d in data} | 
                   {d['database engine 2'] for d in data}), key=sort_key)
  
  escaped_engines = sorted(({d['database engine 1'].replace("_", "\\_") for d in data} | 
                           {d['database engine 2'].replace("_", "\\_") for d in data}), key=sort_key)

  print(engines)
  print(escaped_engines)
  similarity_array = np.zeros((len(engines), len(engines)))  # Initialize for heatmap

  # Create a dictionary to store similarity scores for each engine pair
  similarity_matrix = {}
  for d in data:
    engine1 = d['database engine 1']
    engine2 = d['database engine 2']
    similarity_matrix[(engine1, engine2)] = {
        '2-gram': round(float(d['2-gram similarity']), 2),  # Round to 2 decimals
        '3-gram': round(float(d['3-gram similarity']), 2),
        '4-gram': round(float(d['4-gram similarity']), 2)
    }


    score = round(float(d['4-gram similarity']), 2)
    i = engines.index(engine1)
    j = engines.index(engine2)
    similarity_array[i, j] = score 
    similarity_array[j, i] = score 

  # Generate the LaTeX table
  latex_table = "\\begin{table*}[h]\n"  # Use table*
  latex_table += "\\centering\n"
  latex_table += "\\caption{Similarity Scores Between Database Engines}\n"
  latex_table += "\\begin{tabular}{|c|" + "c" * len(escaped_engines) + "|}\n"
  latex_table += "\\hline\n"
  latex_table += "& " + " & ".join(escaped_engines) + " \\\\\n"
  latex_table += "\\hline\n"

  for engine1 in engines:
    latex_table += engine1.replace("_", "\\_") + " & "
    for engine2 in engines:
      if engine1 == engine2:
        latex_table += "- & "  # Or you can put 1.0000 here
      else:
        scores = similarity_matrix.get((engine1, engine2), similarity_matrix.get((engine2, engine1)))
        if scores:
          latex_table += f"{scores['2-gram']}/{scores['3-gram']}/{scores['4-gram']} & "
        else:
          latex_table += "- & "
    latex_table = latex_table[:-2] + " \\\\\n"  # Remove trailing " & "
  latex_table += "\\hline\n"
  latex_table += "\\end{tabular}\n"
  latex_table += "\\end{table*}"


  # Generate the heatmap
  fig, ax = plt.subplots(figsize=(10, 8))  # Adjust figure size as needed
  im = ax.imshow(similarity_array, cmap='viridis')  # Choose a colormap

  # Show all ticks and label them with the respective list entries
  ax.set_xticks(np.arange(len(engines)), labels=engines)
  ax.set_yticks(np.arange(len(engines)), labels=engines)

  # Rotate the tick labels and set their alignment
  plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
           rotation_mode="anchor")

  ax.set_title("4-gram Similarity Between Database Engines (full dataset)")
  fig.tight_layout()

  plt.colorbar(im)  # Add a colorbar
  plt.savefig("heatmap.png")  # Save the heatmap
  plt.show()



  # Generate the Markdown table
  markdown_table = "| |" + "|".join(engines) + "|\n"
  markdown_table += "|---" * (len(engines) + 1) + "|\n"  # Markdown table header separator

  for engine1 in engines:
    markdown_table += f"| {engine1} |"
    for engine2 in engines:
      if engine1 == engine2:
        markdown_table += " - |"
      else:
        scores = similarity_matrix.get((engine1, engine2), similarity_matrix.get((engine2, engine1)))
        if scores:
          markdown_table += f" {scores['2-gram']}/{scores['3-gram']}/{scores['4-gram']} |"
        else:
          markdown_table += " - |"
    markdown_table += "\n"

  return latex_table, markdown_table

def humanize_db_name(name):
  """Formats a database name in a human-readable way.

  Args:
    name: The database name to format.

  Returns:
    The formatted database name.
  """
  parts = name.split('_')
  if parts[0] == 'postgres':
    if len(parts) == 2:
      return f"PostgreSQL {parts[1]}"
    elif len(parts) == 3:
      if parts[2] == 'gcp':
        return f"PostgreSQL {parts[1]} (GCSQL)"
      else:
        return f"PostgreSQL {parts[1]}.{parts[2]}"
    elif len(parts) == 4:
      return f"PostgreSQL {parts[1]}.{parts[2]} (GCSQL)"
  elif parts[0] == 'sqlserver':
    if len(parts) == 3:
      return f"SQL Server {parts[1]} {parts[2].title()}"
    elif len(parts) == 4:
      return f"SQL Server {parts[1]} {parts[2].title()} (GCSQL)"
  elif parts[0] == 'mysql':
    if len(parts) == 3:
      return f"MySQL {parts[1]}.{parts[2]}"
    elif len(parts) == 4:
      return f"MySQL {parts[1]}.{parts[2]} (GCSQL)"
  return name  # Return the original name if no match is found

def parse_files(folder_path):
  """
  Reads all files in a given folder, parses them, and extracts relevant information.

  Args:
    folder_path: The path to the folder containing the files.

  Returns:
    A list of dictionaries, where each dictionary represents a file and contains
    the extracted information.
  """

  data = []
  for filename in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, filename)):
      with open(os.path.join(folder_path, filename), 'r') as f:
        file_content = f.read()

      # Extract placeholders using regular expressions
      match = re.search(r"(.+)\s+([a-zA-Z0-9_]+)\n\*\*\*\*\*\n2-gram TF-IDF similarity: (.+)\n3-gram TF-IDF similarity: (.+)\n4-gram TF-IDF similarity: (.+)\n\*\*\*\*\*", file_content)
      if match:
        data.append({
            "file": filename,
            "database engine 1": humanize_db_name(match.group(1)),
            "database engine 2": humanize_db_name(match.group(2)),
            "2-gram similarity": match.group(3),
            "3-gram similarity": match.group(4),
            "4-gram similarity": match.group(5)
        })
  return data

if __name__ == "__main__":
  folder_path = "/Users/raresfolea/code/project-martial/packets/results"
  parsed_data = parse_files(folder_path)
  # pprint.pprint(parsed_data)
  latex_code, markdown_table = generate_tables(parsed_data)
  # print(latex_code)
  print(markdown_table)

