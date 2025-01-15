import uuid
import os
import sys
import shutil
import json

    
def transform_path_to_find_in_thecrawlcodeforces(file_path):
    """
    Transforms the given file path according to the specified rules.

    Args:
        file_path (str): The file path to transform.

    Returns:
        str: The transformed file path.
    """

    # Split the path into parts
    parts = file_path.split('/')

    # Remove 'processed' and 'atomic_time'
    parts = [part for part in parts if part not in ('processed', 'atomic_perf', 'atomic_time')]

    # Replace 'TheOutputsCodeforces' with 'TheCrawlCodeforces'
    parts = ['TheCrawlCodeforces' if part == 'TheOutputsCodeforces' else part for part in parts]

    # Extract the last part (filename without extension)
    filename = os.path.splitext(parts[-1])[0]

    # Join the parts back together and add '.CPP' extension
    transformed_path = '/'.join(parts[:-1]) + '.CPP'
    
    return transformed_path


def find_and_read_json(root_path):
    """
    Recursively searches for 'PROCESSED.RAF' files under the given root path,
    reads them as JSON, and stores the JSON data along with the file's absolute path.

    Args:
        root_path (str): The root directory to start the search from.

    Returns:
        list: A list of tuples, where each tuple contains the absolute file path and the loaded JSON data.
    """
    good = 0
    bad = 0
    
    json_data_list = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename == "PROCESSED.RAF":
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r') as f:
                        json_data = json.load(f)
                        thecrawlcodeforces_path = transform_path_to_find_in_thecrawlcodeforces(file_path)
                        if os.path.exists(thecrawlcodeforces_path):
                            good += 1
                        else:
                            bad += 1
                            print(thecrawlcodeforces_path)
                        json_data_list.append((file_path, thecrawlcodeforces_path, json_data))
                except FileNotFoundError:
                    print(f"Warning: File not found: {file_path}")
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON in file: {file_path}")
    
    
    print(f"Found {good} good thecrawlcodeforces_path")
    print(f"Found {bad} bad thecrawlcodeforces_path")
    return json_data_list

def compare_jsons(json1_data, json2_data):
    """
    Compares two JSON-like dictionaries to check if they have at least 6 matching
    FEATURE_TYPE and FEATURE_CONFIG values within their 'metrics' sections.

    Args:
        json1_data: The first JSON-like dictionary.
        json2_data: The second JSON-like dictionary.

    Returns:
        True if at least 6 matches are found, False otherwise.
    """

    metrics1 = json1_data.get('metrics', {})
    metrics2 = json2_data.get('metrics', {})

    matching_count = 0
    for key, value1 in metrics1.items():
        value2 = metrics2.get(key)
        if value2:
            if (value1['FEATURE_TYPE'] == value2['FEATURE_TYPE'] and
                    value1['FEATURE_CONFIG'] == value2['FEATURE_CONFIG']):
                matching_count += 1

    return matching_count >= 6


def parse_second_to_last_part(file_path):
  """
  Parses the second to last part of a file name from a given file path.

  Args:
    file_path: The path to the file.

  Returns:
    The second to last part of the file name, or None if the path is invalid.
  """
  try:
    # Extract the directory and file name
    path, filename = os.path.split(file_path)
    # Split the path into its components
    path_parts = path.split(os.sep)
    # Return the second to last part
    return path_parts[-2]
  except:
    return None



def process_files(processed_data):
    """
    Processes a list of file paths by assigning UUIDs, creating directories,
    copying files, and adding metadata.

    Args:
        file_paths: A list of file paths to process.
    """

    for _, file_path, _ in processed_data:
        # Generate UUID
        file_uuid = str(uuid.uuid4())

        # Create directory
        new_dir = os.path.join('/Users/raf/code/project-martial/dataset/CoSiM/raw', file_uuid)
        os.makedirs(new_dir, exist_ok=True)

        # Copy file
        shutil.copy2(file_path, new_dir)

        # Create metadata
        metadata = {
            "source": file_path,
            "human_readable_source": "TheCrawlCodeforces",
            "import_statement": "via helper script import_from_thecrawlcodeforces.py"
        }

        # Write metadata to file
        metadata_path = os.path.join(new_dir, 'METADATA.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    root_path = "/Users/raf/code/TheOutputsCodeforces/processed/atomic_perf"
    processed_data = find_and_read_json(root_path)
    
    
    similar_same_problem_id = 0
    similar_different_problem_id = 0
    not_similar_same_problem_id = 0
    not_similar_different_problem_id = 0
    
    # Only needed once.
    # process_files(processed_data)
    
    for file_path1, thecrawlcodeforces_path1, json_data1 in processed_data:
        for file_path2, thecrawlcodeforces_path2, json_data2 in processed_data:
            if file_path1 == file_path2:
                continue
            problemid1 = parse_second_to_last_part(file_path1)
            problemid2 = parse_second_to_last_part(file_path2)
            if compare_jsons(json_data1,json_data2):
                if problemid1 == problemid2:
                    similar_same_problem_id += 1
                else:
                    similar_different_problem_id += 1
                # print(file_path1)
                # print(file_path2)
                # print(json_data1)
                # print(json_data2)
            else:
                if problemid1 == problemid2:
                    not_similar_same_problem_id += 1
                else:
                    not_similar_different_problem_id += 1
                
    print(f"Found {similar_same_problem_id} similar_same_problem_id thecrawlcodeforces_path")
    print(f"Found {similar_different_problem_id} similar_different_problem_id thecrawlcodeforces_path")
    print(f"Found {not_similar_same_problem_id} not_similar_same_problem_id thecrawlcodeforces_path")
    print(f"Found {not_similar_different_problem_id} not_similar_different_problem_id thecrawlcodeforces_path")