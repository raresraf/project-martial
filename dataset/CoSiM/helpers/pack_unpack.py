import json
import os

def split_json(input_file, max_size_kb=50000, output_prefix="chunk"):
    """Splits a large JSON file into smaller files based on size.

    Args:
        input_file: Path to the input JSON file.
        max_size_kb: Maximum size of each output chunk in kilobytes.
        output_prefix: Prefix for the output file names (e.g., "chunk").
    """

    try:
        with open(input_file, 'r', encoding='utf-8') as f:  # Handle potential encoding issues
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {input_file}: {e}")
        return

    if not isinstance(data, dict) and not isinstance(data, list):  # Check for valid json structure
        print(f"Invalid JSON format in {input_file}. Must be a dictionary or list.")
        return

    num_chunks = 0
    current_chunk = {} if isinstance(data, dict) else []
    current_size = 0

    for key, value in data.items() if isinstance(data, dict) else enumerate(data):
        item_size = len(json.dumps({key: value} if isinstance(data, dict) else [value]).encode('utf-8'))  # Size in bytes
        item_size_kb = item_size / 1024

        if current_size + item_size_kb > max_size_kb:
            _write_chunk(current_chunk, output_prefix, num_chunks)
            num_chunks += 1
            current_chunk = {} if isinstance(data, dict) else []
            current_size = 0

        if isinstance(data, dict):
          current_chunk[key] = value
        else:
          current_chunk.append(value)
        current_size += item_size_kb


    # Write the last chunk
    if current_chunk:
        _write_chunk(current_chunk, output_prefix, num_chunks)



def _write_chunk(chunk_data, output_prefix, chunk_number):
    output_file = f"{output_prefix}_{chunk_number}.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(chunk_data, outfile, indent=4, ensure_ascii=False)  # Use indent for readability and ensure_ascii=False for unicode
        print(f"Chunk {chunk_number} written to {output_file}")
    except Exception as e:
        print(f"Error writing chunk {chunk_number}: {e}")


def merge_json(input_prefix):
    """Merges JSON files with a given prefix into a single file.

    Args:
        input_prefix: Prefix of the JSON files to merge (e.g., "chunk").
    """
    merged_data = {}  # or [] depending on your original json type
    file_index = 0
    while True:
        input_file = f"{input_prefix}_{file_index}.json"
        if not os.path.exists(input_file):
            break  # No more files to merge

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
              chunk_data = json.load(f)

              if isinstance(chunk_data, dict):
                merged_data.update(chunk_data)
              elif isinstance(chunk_data, list):
                if not isinstance(merged_data, list):
                  merged_data = []
                merged_data.extend(chunk_data)
              else:
                print(f"Invalid json format in chunk {file_index}")
                return

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {input_file}: {e}")
            return
        except Exception as e: # Handle other potential file reading errors
            print(f"Error reading file {input_file}: {e}")
            return

        file_index += 1

    if merged_data:
      output_file = f"{input_prefix}_merged.json"
      try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(merged_data, outfile, indent=4, ensure_ascii=False)
        print(f"Merged data written to {output_file}")
      except Exception as e:
        print(f"Error writing merged file: {e}")
    else:
        print("No files found to merge.")




# Example usage:
# input_file = "notsimdataset.json"
# split_json(input_file, max_size_kb=50000, output_prefix="notsimdataset")
merge_json("notsimdataset")
