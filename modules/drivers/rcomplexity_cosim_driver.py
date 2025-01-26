import re
import os
import sys
import random
import json
from absl import app
from absl import flags
import modules.rcomplexity as rcomplexity
import fcntl
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

dataset = {}
outcome = {}

FLAGS = flags.FLAGS

flags.DEFINE_float("threshold", "0.5", help="Threshold to declare similarity")

flags.DEFINE_float("c11", "0.5061", help="C11")
flags.DEFINE_float("c12", "0.3848", help="C12")
flags.DEFINE_float("c13", "0.0619", help="C13")
flags.DEFINE_float("c14", "0.0", help="C14")

flags.DEFINE_float("c21", "0.3222", help="C21")
flags.DEFINE_float("c22", "0.1076", help="C22")
flags.DEFINE_float("c23", "1.0", help="C23")
flags.DEFINE_float("c24", "0.6837", help="C24")

flags.DEFINE_float("c31", "0.3761", help="C31")
flags.DEFINE_float("c32", "0.4154", help="C32")
flags.DEFINE_float("c33", "0.0", help="C33")
flags.DEFINE_float("c34", "0.0", help="C34")

flags.DEFINE_float("c41", "0.4206", help="C41")
flags.DEFINE_float("c42", "0.6619", help="C42")
flags.DEFINE_float("c43", "0.027", help="C43")
flags.DEFINE_float("c44", "0.0", help="C44")

flags.DEFINE_float("c51", "0.0325", help="C51")
flags.DEFINE_float("c52", "0.2204", help="C52")
flags.DEFINE_float("c53", "0.0195", help="C53")
flags.DEFINE_float("c54", "0.0", help="C54")

flags.DEFINE_float("c61", "0.0", help="C61")
flags.DEFINE_float("c62", "0.1413", help="C62")
flags.DEFINE_float("c63", "0.2458", help="C63")
flags.DEFINE_float("c64", "0.2237", help="C64")

flags.DEFINE_float("c71", "0.3058", help="C71")
flags.DEFINE_float("c72", "0.3331", help="C72")
flags.DEFINE_float("c73", "0.0", help="C73")
flags.DEFINE_float("c74", "0.0", help="C74")

flags.DEFINE_float("c81", "0.0465", help="C81")
flags.DEFINE_float("c82", "0.0631", help="C82")
flags.DEFINE_float("c83", "0.0", help="C83")
flags.DEFINE_float("c84", "0.0", help="C84")

flags.DEFINE_float("c91", "0.1697", help="C91")
flags.DEFINE_float("c92", "0.1808", help="C92")
flags.DEFINE_float("c93", "0.0", help="C93")
flags.DEFINE_float("c94", "0.0", help="C94")

flags.DEFINE_integer("seed", 42, help="seed")


def read_cosim_dataset(filepath):
    """Reads a CoSiM dataset (similar or not similar) from a JSON file.

    Args:
        filepath: The path to the JSON file.

    Returns:
        A list of dictionaries, where each dictionary represents a data point.
        Returns an empty list if there's an error during file reading or JSON decoding.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:  # Explicitly use utf-8 encoding
            data = json.load(f)
            return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing JSON from {filepath}: {e}")
        return []  # Return an empty list to indicate failure
    
def read_cosim_metadata(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing JSON from {filepath}: {e}")
        return []


def convert_path_to_embedding(input_path):
    """Converts a path from TheCrawlCodeforces to TheOutputsCodeforces.

    Args:
        input_path: The original input path.

    Returns:
        The converted output path.
    """

    try:
        # 1. Replace "TheCrawlCodeforces" with "TheOutputsCodeforces/processed/atomic_perf"
        output_path = input_path.replace("TheCrawlCodeforces", "TheOutputsCodeforces/processed/atomic_perf")

        # 2. Remove the ".CPP" extension (case-insensitive)
        if input_path.lower().endswith(".cpp"):
            output_path = output_path[:-4]  # Remove the last 4 characters (".CPP")

        # 3. Add "/PROCESSED.RAF" at the end
        output_path += "/PROCESSED.RAF"

        return output_path

    except Exception as e:  # Catch potential errors (e.g., if the path is malformed)
        print(f"Error processing path: {e}")
        return None


def random_sample_dict(input_dict, sample_size, seed=None):
    """
    Randomly samples a specified number of items from a dictionary with optional seeding for reproducibility.

    Args:
        input_dict: The dictionary to sample from.
        sample_size: The number of items to sample.
        seed: An optional seed value for the random number generator.

    Returns:
        A new dictionary containing the sampled items, or None if the
        input dictionary is empty or the sample size is invalid.
        If the sample size is larger than the dictionary size, it returns a copy of the original dictionary.
    """

    if not input_dict:
        return None

    if sample_size <= 0:
      return {}

    if sample_size >= len(input_dict):
        return input_dict.copy()

    # Seed the random number generator if a seed is provided
    if seed is not None:
        random.seed(seed)

    sampled_keys = random.sample(list(input_dict), sample_size)
    sampled_dict = {key: input_dict[key] for key in sampled_keys}

    return sampled_dict


def run_over_one_dataset(dataset, want_similarity, max_stuff_in_dataset, metadata, rca, gotlst, wantlst):
    max_dataset = random_sample_dict(dataset, max_stuff_in_dataset, seed = FLAGS.seed)
    loss = 0
    for pairid, sim in max_dataset.items():
        if pairid == "0" or pairid == "1":
            continue # Kubernetes
        for uid in sim:
            if not metadata.get(uid, None):
                metadata[uid] = read_cosim_metadata(f"/Users/raf/code/project-martial/dataset/CoSiM/raw/{uid}/METADATA.json")
                metadata[uid]["embedding"] = read_cosim_metadata(convert_path_to_embedding(metadata[uid]["source"]))
            # print(metadata[uid])
        
        if not metadata[sim[0]]["embedding"].get("metrics", None):
            continue
        if not metadata[sim[1]]["embedding"].get("metrics", None):
            continue
        
        rca.fileJSON["file1"] = metadata[sim[0]]["embedding"]
        rca.fileJSON["file2"] = metadata[sim[1]]["embedding"]
        _, _, similarity, _, _, _, _ = (
            rca.find_complexity_similarity_with_as()
        )
        wantlst.append(want_similarity)
        if similarity >= FLAGS.threshold:
            gotlst.append(1)
        else:
            gotlst.append(0)
        loss += -want_similarity * infzerolog(similarity) - (1 - want_similarity) * infzerolog(1 - similarity)
    return loss

def main(_):
    simdataset_path = "/Users/raf/code/project-martial/dataset/CoSiM/similar/simdataset.json"
    notsimdataset_path = "/Users/raf/code/project-martial/dataset/CoSiM/notsimilar/notsimdataset_merged.json"
    
    sim_data = read_cosim_dataset(simdataset_path)
    notsim_data = read_cosim_dataset(notsimdataset_path)
    metadata = {}

    rca = rcomplexity.RComplexityAnalysis()
    rca.disable_find_line = True
    rca.X = [
        [FLAGS.c11, FLAGS.c12, FLAGS.c13, FLAGS.c14],
        [FLAGS.c21, FLAGS.c22, FLAGS.c23, FLAGS.c24],
        [FLAGS.c31, FLAGS.c32, FLAGS.c33, FLAGS.c34],
        [FLAGS.c41, FLAGS.c42, FLAGS.c43, FLAGS.c44],
        [FLAGS.c51, FLAGS.c52, FLAGS.c53, FLAGS.c54],
        [FLAGS.c61, FLAGS.c62, FLAGS.c63, FLAGS.c64],
        [FLAGS.c71, FLAGS.c72, FLAGS.c73, FLAGS.c74],
        [FLAGS.c81, FLAGS.c82, FLAGS.c83, FLAGS.c84],
        [FLAGS.c91, FLAGS.c92, FLAGS.c93, FLAGS.c94],
    ]

    loss = 0

    gotlst = []
    wantlst = []
    max_elems = 90000
    loss += run_over_one_dataset(sim_data, 1, max_elems * 2, metadata, rca, gotlst, wantlst)
    loss += run_over_one_dataset(notsim_data, 0, max_elems, metadata, rca, gotlst, wantlst)
        
    msg = {
        "threshold": FLAGS.threshold,
        "loss": loss,
        # "X": rca.X,
    }
    print(msg)
    print(confusion_matrix(wantlst, gotlst))
    print(classification_report(wantlst, gotlst))
    



def infzerolog(x):
    if x < 3.720076e-44:
        return -100
    return math.log(x)


if __name__ == "__main__":
    app.run(main)


"""
INFO: Running command line: bazel-bin/modules/drivers/rcomplexity_cosim_driver '--seed=42' '--threshold=0.42'
{'threshold': 0.42, 'loss': 129770.17965287855}
[[ 43930  45863]
 [  3251 176747]]
              precision    recall  f1-score   support

           0       0.94      0.50      0.65     89793
           1       0.80      0.98      0.89    179998

    accuracy                           0.83    269791
   macro avg       0.87      0.76      0.77    269791
weighted avg       0.85      0.83      0.81    269791

INFO: Running command line: bazel-bin/modules/drivers/rcomplexity_cosim_driver '--seed=42' '--threshold=0.44'
{'threshold': 0.44, 'loss': 129770.17965287855}
[[ 48454  41339]
 [  5238 174760]]
              precision    recall  f1-score   support

           0       0.91      0.55      0.69     89793
           1       0.82      0.97      0.89    179998

    accuracy                           0.83    269791
   macro avg       0.87      0.77      0.79    269791
weighted avg       0.85      0.84      0.82    269791

INFO: Running command line: bazel-bin/modules/drivers/rcomplexity_cosim_driver '--seed=42' '--threshold=0.46'
{'threshold': 0.46, 'loss': 129770.17965287855}
[[ 52596  37197]
 [ 12151 167847]]
              precision    recall  f1-score   support

           0       0.82      0.60      0.69     89793
           1       0.83      0.94      0.88    179998

    accuracy                           0.83    269791
   macro avg       0.83      0.77      0.79    269791
weighted avg       0.83      0.82      0.82    269791

INFO: Running command line: bazel-bin/modules/drivers/rcomplexity_cosim_driver '--seed=42' '--threshold=0.48'
{'threshold': 0.48, 'loss': 129770.17965287855}
[[ 56481  33312]
 [ 15616 164382]]
              precision    recall  f1-score   support

           0       0.79      0.64      0.71     89793
           1       0.84      0.92      0.88    179998

    accuracy                           0.83    269791
   macro avg       0.82      0.78      0.79    269791
weighted avg       0.82      0.83      0.82    269791

INFO: Running command line: bazel-bin/modules/drivers/rcomplexity_cosim_driver '--seed=42'
{'threshold': 0.5, 'loss': 129770.17965287855}
[[ 60253  29540]
 [ 20578 159420]]
              precision    recall  f1-score   support

           0       0.76      0.68      0.72     89793
           1       0.85      0.90      0.88    179998

    accuracy                           0.83    269791
   macro avg       0.80      0.79      0.80    269791
weighted avg       0.82      0.83      0.82    269791

INFO: Running command line: bazel-bin/modules/drivers/rcomplexity_cosim_driver '--seed=42' '--threshold=0.52'
{'threshold': 0.52, 'loss': 129770.17965287855}
[[ 63649  26144]
 [ 26135 153863]]
              precision    recall  f1-score   support

           0       0.72      0.72      0.72     89793
           1       0.86      0.86      0.86    179998

    accuracy                           0.82    269791
   macro avg       0.79      0.78      0.78    269791
weighted avg       0.82      0.82      0.82    269791

INFO: Running command line: bazel-bin/modules/drivers/rcomplexity_cosim_driver '--seed=42' '--threshold=0.54'
{'threshold': 0.54, 'loss': 129770.17965287855}
[[ 66849  22944]
 [ 32944 147054]]
              precision    recall  f1-score   support

           0       0.68      0.75      0.72     89793
           1       0.88      0.83      0.85    179998

    accuracy                           0.80    269791
   macro avg       0.78      0.79      0.78    269791
weighted avg       0.81      0.79      0.80    269791

INFO: Running command line: bazel-bin/modules/drivers/rcomplexity_cosim_driver '--seed=42' '--threshold=0.56'
{'threshold': 0.56, 'loss': 129770.17965287855}
[[ 69832  19961]
 [ 40006 139992]]
              precision    recall  f1-score   support

           0       0.65      0.79      0.71     89793
           1       0.89      0.79      0.83    179998

    accuracy                           0.79    269791
   macro avg       0.77      0.79      0.77    269791
weighted avg       0.81      0.79      0.79    269791

INFO: Running command line: bazel-bin/modules/drivers/rcomplexity_cosim_driver '--seed=42' '--threshold=0.58'
{'threshold': 0.58, 'loss': 129770.17965287855}
[[ 72448  17345]
 [ 47330 132668]]
              precision    recall  f1-score   support

           0       0.61      0.82      0.70     89793
           1       0.89      0.75      0.82    179998

    accuracy                           0.77    269791
   macro avg       0.75      0.77      0.76    269791
weighted avg       0.80      0.77      0.78    269791
"""