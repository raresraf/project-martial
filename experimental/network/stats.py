"""
\begin{table}[h]
\centering
\begin{tabular}{|l|c|}
\hline
Directory & Size (KB) \\
\hline
mysql_5_6 & 643.60 \\
mysql_5_6_gcp & 388.11 \\
mysql_5_7 & 637.97 \\
mysql_5_7_gcp & 392.43 \\
mysql_8_0 & 662.57 \\
mysql_8_0_gcp & 404.17 \\
postgres_10 & 762.42 \\
postgres_10_gcp & 363.93 \\
postgres_11 & 762.38 \\
postgres_11_gcp & 364.24 \\
postgres_12 & 763.57 \\
postgres_12_gcp & 363.70 \\
postgres_13 & 763.53 \\
postgres_13_gcp & 364.04 \\
postgres_14 & 1173.35 \\
postgres_14_gcp & 610.62 \\
postgres_15 & 1071.92 \\
postgres_15_gcp & 582.39 \\
postgres_16 & 1202.90 \\
postgres_16_gcp & 596.66 \\
postgres_9_6 & 727.00 \\
postgres_9_6_gcp & 364.36 \\
sqlserver_2017_dev & 1125.92 \\
sqlserver_2017_enterprise_gcp & 609.98 \\
sqlserver_2017_standard_gcp & 609.95 \\
sqlserver_2019_dev & 1125.55 \\
sqlserver_2019_enterprise_gcp & 613.46 \\
sqlserver_2019_standard_gcp & 608.84 \\
sqlserver_2022_dev & 1125.56 \\
sqlserver_2022_enterprise_gcp & 609.96 \\
sqlserver_2022_standard_gcp & 610.10 \\
\hline
\end{tabular}
\caption{Directory Sizes}
\label{tab:dir_sizes}
\end{table}
"""

import os

def get_directory_size(directory):
  """
  Calculates the total size of a directory (including subdirectories) in bytes.

  Args:
    directory: The path to the directory.

  Returns:
    The size of the directory in bytes.
  """
  total_size = 0
  for dirpath, dirnames, filenames in os.walk(directory):
    for f in filenames:
      fp = os.path.join(dirpath, f)
      # skip if it is symbolic link
      if not os.path.islink(fp):
        total_size += os.path.getsize(fp)
  return total_size

def main():
  """
  Creates a LaTeX table of directory names and their sizes in kilobytes 
  for the current path, sorted by directory name.
  """
  current_path = os.getcwd() 
  directories = [
      name for name in os.listdir(current_path) 
      if os.path.isdir(os.path.join(current_path, name))
  ]
  directories.sort() 

  print("\\begin{table}[h]")
  print("\\centering")
  print("\\begin{tabular}{|l|c|}")
  print("\\hline")
  print("Directory & Size (KB) \\\\")  # Changed column header
  print("\\hline")
  for directory in directories:
    size_bytes = get_directory_size(directory)
    size_kb = size_bytes / 1024  # Convert bytes to kilobytes
    print(f"{directory} & {size_kb:.2f} \\\\")  # Format to 2 decimal places
  print("\\hline")
  print("\\end{tabular}")
  print("\\caption{Directory Sizes}")
  print("\\label{tab:dir_sizes}")
  print("\\end{table}")

if __name__ == "__main__":
  main()
