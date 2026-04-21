"""A driver function to run similarity experiments on the CoSiM dataset using Gemini-annotated comments.

This driver evaluates the similarity of computer programs by analyzing NLP comments
annotated by Gemini. It uses the Universal Sentence Encoder (USE) to compute
embeddings for sequences of comments and identifies several similarity metrics.
It uses ProcessPoolExecutor for true parallelism and implements robust checkpointing.
"""

import os
import json
import re
import threading
import concurrent.futures
import signal
import sys
import tempfile
from absl import app
from absl import flags
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from modules.comments import CommentsAnalysis
import modules.comments_config as comments_config
import modules.comments_helpers as comments_helpers

FLAGS = flags.FLAGS
flags.DEFINE_string("simdataset", "dataset/CoSiM/similar/simdataset.json", "Path to simdataset.json")
flags.DEFINE_string("notsimdataset", "dataset/CoSiM/notsimilar/notsimdataset.json", "Path to notsimdataset.json")
flags.DEFINE_string("commented_dir", "dataset/CoSiM-commented-by-Gemini/raw/", "Path to commented files")
flags.DEFINE_string("checkpoint", "comments_cosim_checkpoint.json", "Checkpoint file for persistence")
flags.DEFINE_string("output", "comments_cosim_results.json", "Output file for the full results")
flags.DEFINE_integer("limit", 1000000, "Limit number of pairs to process")
flags.DEFINE_integer("workers", 4, "Number of parallel worker processes")
flags.DEFINE_bool("use_smaller_sample", False, "Use only a small sample for testing (e.g., 10 pairs from each)")

# Global results dictionary and its lock for thread-safe updates in the main process
global_results = {}
results_lock = threading.Lock()

def atomic_save(data, filepath):
    """Saves data to a JSON file atomically using a temporary file."""
    fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(filepath))
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f)
        os.replace(temp_path, filepath)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"Error during atomic save: {e}")

def signal_handler(sig, frame):
    """Handles interruption signals to save current progress before exiting."""
    print("\nInterruption received. Saving current progress...")
    with results_lock:
        atomic_save(global_results, FLAGS.checkpoint)
    print("Progress saved. Exiting.")
    sys.exit(0)

def extract_comments_regex(text):
    """Extracts comments from source code using a language-agnostic regex-based approach."""
    findings = []
    # Match // comments
    for match in re.finditer(r'//(.*)', text):
        comment = match.group(1).strip()
        line_num = text.count('\n', 0, match.start()) + 1
        if comment:
            findings.append((comment, line_num))
            
    # Match # comments
    for match in re.finditer(r'(?m)^\s*#(?!include|define|if|else|endif|pragma|import)(.*)', text):
        comment = match.group(1).strip()
        line_num = text.count('\n', 0, match.start()) + 1
        if comment:
            findings.append((comment, line_num))
    
    # Match /* */ block comments
    for match in re.finditer(r'/\*(.*?)\*/', text, re.DOTALL):
        comment_text = match.group(1)
        start_pos = match.start()
        current_line = text.count('\n', 0, start_pos) + 1
        for line in comment_text.split('\n'):
            stripped = line.strip().lstrip('*').strip()
            if stripped:
                findings.append((stripped, current_line))
            current_line += 1
            
    # Python docstrings
    for match in re.finditer(r"'''(.*?)'''", text, re.DOTALL):
        comment_text = match.group(1)
        start_pos = match.start()
        current_line = text.count('\n', 0, start_pos) + 1
        for line in comment_text.split('\n'):
            stripped = line.strip()
            if stripped:
                findings.append((stripped, current_line))
            current_line += 1
            
    for match in re.finditer(r'"""(.*?)"""', text, re.DOTALL):
        comment_text = match.group(1)
        start_pos = match.start()
        current_line = text.count('\n', 0, start_pos) + 1
        for line in comment_text.split('\n'):
            stripped = line.strip()
            if stripped:
                findings.append((stripped, current_line))
            current_line += 1

    return findings

def is_uuid_processed(uuid, base_dir):
    """Checks if a UUID directory contains a .checkpoint file."""
    uuid_dir = os.path.join(base_dir, uuid)
    if not os.path.isdir(uuid_dir):
        return False
    return os.path.exists(os.path.join(uuid_dir, ".checkpoint"))

def get_all_comments_for_uuid(uuid, base_dir):
    uuid_dir = os.path.join(base_dir, uuid)
    if not os.path.isdir(uuid_dir):
        return []
    
    all_findings = []
    for root, _, files in os.walk(uuid_dir):
        for file in files:
            if file == "METADATA.json" or file.endswith(".checkpoint"):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    all_findings.extend(extract_comments_regex(content))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return all_findings

worker_ca = None

def init_worker():
    """Initializes the CommentsAnalysis object for the worker process."""
    global worker_ca
    import modules.comments_config as worker_config
    worker_config.config.set_enable_use(True)
    from modules.comments import CommentsAnalysis
    worker_ca = CommentsAnalysis()

def process_single_pair(pair_id, pair_uuids, label, commented_dir):
    """Processes a single pair of UUIDs. This runs in a worker process."""
    global worker_ca
    
    u1, u2 = pair_uuids
    if not is_uuid_processed(u1, commented_dir) or not is_uuid_processed(u2, commented_dir):
        return None, None

    findings1 = get_all_comments_for_uuid(u1, commented_dir)
    findings2 = get_all_comments_for_uuid(u2, commented_dir)
    
    if not findings1 or not findings2:
        return None, None
    
    max_sim = 0.0
    avg_best_match_sim = 0.0
    holistic_sim = 0.0
    coverage_sim = 0.0
    threshold = 0.8

    seq1 = worker_ca.comm_to_seq_use(findings1)
    seq2 = worker_ca.comm_to_seq_use(findings2)
    
    if seq1 and seq2:
        emb1 = [s[2].numpy().reshape(512) for s in seq1]
        emb2 = [s[2].numpy().reshape(512) for s in seq2]
        arr1 = np.vstack(emb1)
        arr2 = np.vstack(emb2)
        sim_matrix = cosine_similarity(arr1, arr2)
        
        max_sim = float(np.max(sim_matrix))
        best_matches_A = np.max(sim_matrix, axis=1)
        avg_best_match_sim = float(np.mean(best_matches_A))
        
        mean1 = np.mean(arr1, axis=0).reshape(1, -1)
        mean2 = np.mean(arr2, axis=0).reshape(1, -1)
        holistic_sim = float(cosine_similarity(mean1, mean2)[0][0])

        from modules.comments_helpers import generate_comm_sequences
        indices_seq1 = generate_comm_sequences(range(len(findings1)), 6)
        indices_seq2 = generate_comm_sequences(range(len(findings2)), 6)
        
        covered_indices_1 = [False] * len(findings1)
        for i, best_sim in enumerate(best_matches_A):
            if best_sim >= threshold:
                for idx in indices_seq1[i]:
                    covered_indices_1[idx] = True
        coverage_1 = sum(covered_indices_1) / len(findings1)
        
        best_matches_B = np.max(sim_matrix, axis=0)
        covered_indices_2 = [False] * len(findings2)
        for j, best_sim in enumerate(best_matches_B):
            if best_sim >= threshold:
                for idx in indices_seq2[j]:
                    covered_indices_2[idx] = True
        coverage_2 = sum(covered_indices_2) / len(findings2)
        coverage_sim = (coverage_1 + coverage_2) / 2.0
    
    res_key = f"{label}_{pair_id}"
    result_data = {
        "pair_id": pair_id,
        "uuids": pair_uuids,
        "label": label,
        "max_similarity": max_sim,
        "avg_best_match_similarity": avg_best_match_sim,
        "holistic_similarity": holistic_sim,
        "coverage_similarity": coverage_sim
    }
    return res_key, result_data

def main(_):
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load datasets
    print(f"Loading datasets from {FLAGS.simdataset} and {FLAGS.notsimdataset}...")
    try:
        sim_data = {}
        if os.path.exists(FLAGS.simdataset):
            with open(FLAGS.simdataset, 'r') as f:
                sim_data = json.load(f)
        
        notsim_data = {}
        if os.path.exists(FLAGS.notsimdataset):
            with open(FLAGS.notsimdataset, 'r') as f:
                notsim_data = json.load(f)
        else:
            import subprocess
            try:
                out = subprocess.check_output(["cat", FLAGS.notsimdataset])
                notsim_data = json.loads(out)
            except:
                notsim_data = {}
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
    
    # Load checkpoint into global results
    global global_results
    if os.path.exists(FLAGS.checkpoint):
        try:
            with open(FLAGS.checkpoint, 'r') as f:
                global_results = json.load(f)
            print(f"Loaded {len(global_results)} results from checkpoint.")
        except:
            print("Checkpoint corrupted or empty, starting fresh.")
    
    pairs_to_process = []
    if FLAGS.use_smaller_sample:
        sim_keys = list(sim_data.keys())[:10]
        for k in sim_keys:
            pairs_to_process.append((k, sim_data[k], 1))
        notsim_keys = list(notsim_data.keys())[:10]
        for k in notsim_keys:
            pairs_to_process.append((k, notsim_data[k], 0))
    else:
        for k, v in sim_data.items():
            pairs_to_process.append((k, v, 1))
        for k, v in notsim_data.items():
            pairs_to_process.append((k, v, 0))
        
    total = min(len(pairs_to_process), FLAGS.limit)
    pairs_to_process = pairs_to_process[:total]
    
    # Filter out already processed pairs
    remaining_pairs = [p for p in pairs_to_process if f"{p[2]}_{p[0]}" not in global_results]
    print(f"Total pairs to evaluate: {total}. Remaining: {len(remaining_pairs)}")
    
    if not remaining_pairs:
        print("No new pairs to process.")
    else:
        print(f"Starting parallel processing with {FLAGS.workers} worker processes...")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=FLAGS.workers, 
            initializer=init_worker
        ) as executor:
            future_to_pair = {
                executor.submit(process_single_pair, pid, puuids, lbl, FLAGS.commented_dir): (pid, puuids, lbl)
                for pid, puuids, lbl in remaining_pairs
            }
            
            for future in concurrent.futures.as_completed(future_to_pair):
                try:
                    res_key, result_data = future.result()
                    if res_key:
                        with results_lock:
                            global_results[res_key] = result_data
                            count_finished = len(global_results)
                            if count_finished % 10 == 0:
                                print(f"Progress: {count_finished}/{total} (Latest: {res_key}, Sim: {result_data['coverage_similarity']:.4f})")
                            if count_finished % 100 == 0:
                                atomic_save(global_results, FLAGS.checkpoint)
                except Exception as e:
                    pair = future_to_pair[future]
                    print(f"Error processing pair {pair}: {e}")

    # Final save and analysis
    print(f"\nSaving final results to {FLAGS.output}")
    save_results_and_analyze(global_results, FLAGS.output)
    with results_lock:
        atomic_save(global_results, FLAGS.checkpoint)
    print("Done!")

def save_results_and_analyze(results, output_path):
    from sklearn.metrics import confusion_matrix, classification_report
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    if not results: return
    y_true = [r['label'] for r in results.values()]
    metrics = ['max_similarity', 'avg_best_match_similarity', 'holistic_similarity', 'coverage_similarity']
    
    analysis = {}
    print("\n--- Model Performance Analysis ---")
    for m in metrics:
        print(f"\nMetric: {m}")
        y_scores = [r.get(m, 0.0) for r in results.values()]
        y_pred = [1 if s >= 0.8 else 0 for s in y_scores]
        try:
            print(confusion_matrix(y_true, y_pred))
            print(classification_report(y_true, y_pred))
            analysis[m] = classification_report(y_true, y_pred, output_dict=True)
        except: pass

    with open(output_path.replace(".json", "_summary.json"), "w") as f:
        json.dump(analysis, f, indent=4)

if __name__ == "__main__":
    app.run(main)
