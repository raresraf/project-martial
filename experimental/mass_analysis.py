import subprocess
import sys
import time
import threading


MAX_PROCESSES = 6 

def check_file_length(filename):
  try:
    with open(filename, 'r') as f:
      lines = f.readlines()
      return len(lines) >= 9
  except FileNotFoundError:
    return False

def run_comparison(f1, f2):
    """Runs the comparison script as a subprocess with the given arguments."""
    filename = f"results/output_{f1}_{f2}.txt"
    if check_file_length(filename):
        return
    with open(filename, "w") as outfile:
        subprocess.run(["python3", "tfidf.py", f1, f2], stdout=outfile)


if __name__ == "__main__":
    dbs = [
        "postgres_9_6",
        "postgres_9_6_gcp",
        "postgres_10",
        "postgres_10_gcp",
        "postgres_11",
        "postgres_11_gcp",
        "postgres_12",
        "postgres_12_gcp",
        "postgres_13",
        "postgres_13_gcp",
        "postgres_14",
        "postgres_14_gcp",
        "postgres_15",
        "postgres_15_gcp",
        "postgres_16",
        "postgres_16_gcp",
        "sqlserver_2017_dev",
        "sqlserver_2017_standard_gcp",
        "sqlserver_2017_enterprise_gcp",
        "sqlserver_2019_dev",
        "sqlserver_2019_standard_gcp",
        "sqlserver_2019_enterprise_gcp",
        "sqlserver_2022_dev",
        "sqlserver_2022_standard_gcp",
        "sqlserver_2022_enterprise_gcp",
        "mysql_5_6",
        "mysql_5_6_gcp",
        "mysql_5_7",
        "mysql_5_7_gcp",
        "mysql_8_0",
        "mysql_8_0_gcp",
    ]
    
    threads = []
    for f1 in dbs:
        for f2 in dbs:
            if f1 > f2:
                thread = threading.Thread(target=run_comparison, args=(f1, f2))
                threads.append(thread)
                thread.start()
                print(f"({f1}, {f2}) has started")

                while threading.active_count() > MAX_PROCESSES:
                    time.sleep(1)  # Wait for a second and check again

    for thread in threads:
        thread.join()  # Wait for all threads to finish
