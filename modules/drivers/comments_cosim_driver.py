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
import time
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
flags.DEFINE_string("checkpoint", "results_cosim/checkpoint.json", "Checkpoint file for persistence")
flags.DEFINE_string("output", "results_cosim/results.json", "Output file for the full results")
flags.DEFINE_integer("limit", 1000000, "Limit number of pairs to process")
flags.DEFINE_integer("workers", 4, "Number of parallel worker processes")
flags.DEFINE_bool("use_smaller_sample", False, "Use only a small sample for testing (e.g., 10 pairs from each)")
flags.DEFINE_float("threshold", 0.5, "Similarity threshold for detection")
flags.DEFINE_string("embeddings_cache_dir", "results_cosim/embeddings_cache", "Directory to cache USE embeddings")

# Global results dictionary and its lock for thread-safe updates in the main process
global_results = {}
results_lock = threading.Lock()

def get_cached_embeddings(uuid, commented_dir, cache_dir, worker_ca):
    """Retrieves embeddings from cache or computes and saves them."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    cache_path = os.path.join(cache_dir, f"{uuid}_use_r6.npz")
    
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            # Reconstruct the expected format: list of (long_comm, coming_from, embedding_tensor)
            # Note: we store embeddings as numpy, and reconstruct them or just return numpy
            long_comms = data['long_comms']
            coming_froms = data['coming_froms']
            embeddings = data['embeddings']
            
            # Since the script uses .numpy() on the result of comm_to_seq_use, 
            # we can return a simplified structure or the same structure.
            # To keep process_single_pair mostly unchanged, we'll return the same structure.
            # However, the worker_ca.use returns tensors. We'll store/return numpy arrays.
            res = []
            for i in range(len(long_comms)):
                res.append((long_comms[i], coming_froms[i], embeddings[i]))
            return res
        except Exception as e:
            print(f"Error loading cache for {uuid}: {e}")

    # Not in cache, compute it
    findings = get_all_comments_for_uuid(uuid, commented_dir)
    if not findings:
        return []
    
    seq = worker_ca.comm_to_seq_use(findings, t=6)
    if not seq:
        return []

    # Prepare for saving
    long_comms = [s[0] for s in seq]
    coming_froms = [s[1] for s in seq]
    # Convert tensors to numpy
    embeddings = [s[2].numpy().reshape(512) for s in seq]
    
    try:
        # Atomic-ish save with numpy
        temp_fd, temp_path = tempfile.mkstemp(dir=cache_dir)
        os.close(temp_fd)
        np.savez_compressed(temp_path, long_comms=long_comms, coming_froms=coming_froms, embeddings=embeddings)
        os.replace(temp_path, cache_path)
    except Exception as e:
        print(f"Error saving cache for {uuid}: {e}")
        
    # Return in the same format (using numpy arrays for embeddings)
    res = []
    for i in range(len(long_comms)):
        res.append((long_comms[i], coming_froms[i], embeddings[i]))
    return res

def atomic_save(data, filepath):
    """Saves data to a JSON file atomically using a temporary file."""
    parent = os.path.dirname(filepath)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=parent if parent else None)
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

def process_single_pair(pair_id, pair_uuids, label, commented_dir, threshold, cache_dir):
    """Processes a single pair of UUIDs. This runs in a worker process."""
    global worker_ca
    
    u1, u2 = pair_uuids
    print(f"[{time.strftime('%H:%M:%S')}] Worker starting pair {pair_id} ({u1}, {u2})")
    
    if not is_uuid_processed(u1, commented_dir) or not is_uuid_processed(u2, commented_dir):
        print(f"[{time.strftime('%H:%M:%S')}] Pair {pair_id} skipped: UUIDs not processed")
        return None, None

    # Load from cache or compute
    seq1 = get_cached_embeddings(u1, commented_dir, cache_dir, worker_ca)
    seq2 = get_cached_embeddings(u2, commented_dir, cache_dir, worker_ca)
    
    if not seq1 or not seq2:
        print(f"[{time.strftime('%H:%M:%S')}] Pair {pair_id} skipped: No comments found (F1: {len(seq1)}, F2: {len(seq2)})")
        return None, None
    
    print(f"[{time.strftime('%H:%M:%S')}] Pair {pair_id} has {len(seq1)} and {len(seq2)} comment sequences.")
    
    coverage_sim_6 = 0.0
    coverage_sim_3 = 0.0
    coverage_sim_1 = 0.0
    coverage_sim = 0.0

    if seq1 and seq2:
        # embeddings in seq are already numpy arrays from get_cached_embeddings
        emb1 = [s[2] for s in seq1]
        emb2 = [s[2] for s in seq2]
        arr1 = np.vstack(emb1)
        arr2 = np.vstack(emb2)
        sim_matrix = cosine_similarity(arr1, arr2)
        
        best_matches_A = np.max(sim_matrix, axis=1)
        best_matches_B = np.max(sim_matrix, axis=0)

        # To calculate coverage, we need original findings counts
        findings1 = get_all_comments_for_uuid(u1, commented_dir)
        findings2 = get_all_comments_for_uuid(u2, commented_dir)
        
        from modules.comments_helpers import generate_comm_sequences
        indices_seq1_6 = generate_comm_sequences(range(len(findings1)), 6)
        indices_seq2_6 = generate_comm_sequences(range(len(findings2)), 6)
        
        # Track coverage for each line individually to compute the union
        total_covered_1 = [False] * len(findings1)
        total_covered_2 = [False] * len(findings2)

        # Coverage for r=6
        covered_indices_1_6 = [False] * len(findings1)
        for i, best_sim in enumerate(best_matches_A):
            if best_sim >= threshold:
                for idx in indices_seq1_6[i]:
                    covered_indices_1_6[idx] = True
                    total_covered_1[idx] = True
        coverage_1_6 = sum(covered_indices_1_6) / len(findings1)
        
        covered_indices_2_6 = [False] * len(findings2)
        for j, best_sim in enumerate(best_matches_B):
            if best_sim >= threshold:
                for idx in indices_seq2_6[j]:
                    covered_indices_2_6[idx] = True
                    total_covered_2[idx] = True
        coverage_2_6 = sum(covered_indices_2_6) / len(findings2)
        coverage_sim_6 = (coverage_1_6 + coverage_2_6) / 2.0

        # Coverage for r=3
        mask1_3 = [len(idx_tuple) <= 3 for idx_tuple in indices_seq1_6]
        mask2_3 = [len(idx_tuple) <= 3 for idx_tuple in indices_seq2_6]
        
        if any(mask1_3) and any(mask2_3):
            sim_matrix_3 = sim_matrix[np.ix_(mask1_3, mask2_3)]
            best_matches_A_3 = np.max(sim_matrix_3, axis=1)
            best_matches_B_3 = np.max(sim_matrix_3, axis=0)
            
            indices_seq1_3_only = [idx_tuple for idx_tuple in indices_seq1_6 if len(idx_tuple) <= 3]
            indices_seq2_3_only = [idx_tuple for idx_tuple in indices_seq2_6 if len(idx_tuple) <= 3]
            
            covered_indices_1_3 = [False] * len(findings1)
            for i, best_sim in enumerate(best_matches_A_3):
                if best_sim >= threshold:
                    for idx in indices_seq1_3_only[i]:
                        covered_indices_1_3[idx] = True
                        total_covered_1[idx] = True
            coverage_1_3 = sum(covered_indices_1_3) / len(findings1)
            
            covered_indices_2_3 = [False] * len(findings2)
            for j, best_sim in enumerate(best_matches_B_3):
                if best_sim >= threshold:
                    for idx in indices_seq2_3_only[j]:
                        covered_indices_2_3[idx] = True
                        total_covered_2[idx] = True
            coverage_2_3 = sum(covered_indices_2_3) / len(findings2)
            coverage_sim_3 = (coverage_1_3 + coverage_2_3) / 2.0

        # Coverage for r=1
        mask1_1 = [len(idx_tuple) == 1 for idx_tuple in indices_seq1_6]
        mask2_1 = [len(idx_tuple) == 1 for idx_tuple in indices_seq2_6]
        
        if any(mask1_1) and any(mask2_1):
            sim_matrix_1 = sim_matrix[np.ix_(mask1_1, mask2_1)]
            best_matches_A_1 = np.max(sim_matrix_1, axis=1)
            best_matches_B_1 = np.max(sim_matrix_1, axis=0)
            
            indices_seq1_1_only = [idx_tuple for idx_tuple in indices_seq1_6 if len(idx_tuple) == 1]
            indices_seq2_1_only = [idx_tuple for idx_tuple in indices_seq2_6 if len(idx_tuple) == 1]
            
            covered_indices_1_1 = [False] * len(findings1)
            for i, best_sim in enumerate(best_matches_A_1):
                if best_sim >= threshold:
                    for idx in indices_seq1_1_only[i]:
                        covered_indices_1_1[idx] = True
                        total_covered_1[idx] = True
            coverage_1_1 = sum(covered_indices_1_1) / len(findings1)
            
            covered_indices_2_1 = [False] * len(findings2)
            for j, best_sim in enumerate(best_matches_B_1):
                if best_sim >= threshold:
                    for idx in indices_seq2_1_only[j]:
                        covered_indices_2_1[idx] = True
                        total_covered_2[idx] = True
            coverage_2_1 = sum(covered_indices_2_1) / len(findings2)
            coverage_sim_1 = (coverage_1_1 + coverage_2_1) / 2.0

        # Combined Coverage (Union of r=1, 3, 6)
        coverage_1_combined = sum(total_covered_1) / len(findings1)
        coverage_2_combined = sum(total_covered_2) / len(findings2)
        coverage_sim = (coverage_1_combined + coverage_2_combined) / 2.0
    
    print(f"[{time.strftime('%H:%M:%S')}] Worker finished pair {pair_id}. Sim (Combined): {coverage_sim:.4f}, Sim (r=6): {coverage_sim_6:.4f}, Sim (r=3): {coverage_sim_3:.4f}, Sim (r=1): {coverage_sim_1:.4f}")
    res_key = f"{label}_{pair_id}"
    result_data = {
        "pair_id": pair_id,
        "uuids": pair_uuids,
        "label": label,
        "coverage_similarity": coverage_sim,
        "coverage_similarity_6": coverage_sim_6,
        "coverage_similarity_3": coverage_sim_3,
        "coverage_similarity_1": coverage_sim_1
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
        start_time = time.time()
        num_to_process = len(remaining_pairs)
        processed_this_run = 0

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=FLAGS.workers, 
            initializer=init_worker
        ) as executor:
            future_to_pair = {
                executor.submit(process_single_pair, pid, puuids, lbl, FLAGS.commented_dir, FLAGS.threshold, FLAGS.embeddings_cache_dir): (pid, puuids, lbl)
                for pid, puuids, lbl in remaining_pairs
            }
            
            for future in concurrent.futures.as_completed(future_to_pair):
                try:
                    res_key, result_data = future.result()
                    processed_this_run += 1
                    if res_key:
                        with results_lock:
                            global_results[res_key] = result_data
                            count_finished = len(global_results)
                            
                            elapsed = time.time() - start_time
                            speed = processed_this_run / elapsed if elapsed > 0 else 0
                            remaining = num_to_process - processed_this_run
                            eta = remaining / speed if speed > 0 else 0
                            
                            if processed_this_run % 1 == 0: # Log every pair for better visibility
                                print(f"[{time.strftime('%H:%M:%S')}] Progress: {count_finished}/{total} "
                                      f"({processed_this_run}/{num_to_process} this run) | "
                                      f"Speed: {speed:.2f} pairs/s | ETA: {eta:.1f}s | "
                                      f"Latest: {res_key} (Sim: {result_data['coverage_similarity']:.4f})")
                            
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
    metrics = ['coverage_similarity', 'coverage_similarity_6', 'coverage_similarity_3', 'coverage_similarity_1']
    
    analysis = {}
    print("\n--- Model Performance Analysis ---")
    for m in metrics:
        print(f"\nMetric: {m}")
        y_scores = [r.get(m, 0.0) for r in results.values()]
        y_pred = [1 if s >= FLAGS.threshold else 0 for s in y_scores]
        try:
            print(confusion_matrix(y_true, y_pred))
            print(classification_report(y_true, y_pred))
            analysis[m] = classification_report(y_true, y_pred, output_dict=True)
        except: pass

    with open(output_path.replace(".json", "_summary.json"), "w") as f:
        json.dump(analysis, f, indent=4)

if __name__ == "__main__":
    app.run(main)
