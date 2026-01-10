import os
import re
from bs4 import BeautifulSoup
import numpy as np
import csv
import argparse

def parse_moss_report(file_path):
    """Parses a MOSS report to extract similarity percentages, ignoring same-source matches based on UUID."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except (IOError, OSError):
        return None

    if "No matches were found in your submission." in html_content:
        return 0

    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    if not table:
        return 0

    rows = table.find_all('tr')
    if len(rows) <= 1:
        return 0

    max_valid_similarity = 0

    for row in rows[1:]:
        cols = row.find_all('td')
        if len(cols) < 2:
            continue

        text1_tag = cols[0].find('a')
        text2_tag = cols[1].find('a')

        if not text1_tag or not text2_tag:
            continue

        text1 = text1_tag.get_text().strip()
        text2 = text2_tag.get_text().strip()

        path1_raw = re.sub(r'\s*\(\d+%\)$', '', text1).strip()
        path2_raw = re.sub(r'\s*\(\d+%\)$', '', text2).strip()

        path1_components = path1_raw.split('/')
        path2_components = path2_raw.split('/')

        if len(path1_components) < 2 or len(path2_components) < 2:
            continue

        uuid1 = path1_components[1]
        uuid2 = path2_components[1]

        if uuid1 == uuid2:
            continue

        p1_match = re.search(r'\((\d+)%\)', text1)
        p2_match = re.search(r'\((\d+)%\)', text2)

        p1 = int(p1_match.group(1)) if p1_match else 0
        p2 = int(p2_match.group(1)) if p2_match else 0

        max_valid_similarity = max(max_valid_similarity, p1, p2)

    return max_valid_similarity

def debug_evaluation(sim_matches, not_sim_matches, debug_threshold):
    """Prints the indices of pairs for TP, FN, FP, TN for a given threshold."""
    print(f"\n--- Debug Mode: Threshold @ {debug_threshold}% ---")

    tp_indices = [k for k, v in sim_matches.items() if v > debug_threshold]
    fn_indices = [k for k, v in sim_matches.items() if v <= debug_threshold]
    fp_indices = [k for k, v in not_sim_matches.items() if v > debug_threshold]
    tn_indices = [k for k, v in not_sim_matches.items() if v <= debug_threshold]

    print(f"\nTrue Positives (TP): {len(tp_indices)} pairs")
    print(tp_indices)

    print(f"\nFalse Negatives (FN): {len(fn_indices)} pairs")
    print(fn_indices)

    print(f"\nFalse Positives (FP): {len(fp_indices)} pairs")
    print(fp_indices)

    print(f"\nTrue Negatives (TN): {len(tn_indices)} pairs")
    print(tn_indices)
    print("\n--- End Debug Mode ---")

def main():
    """Main function to evaluate MOSS reports and generate CSV."""
    parser = argparse.ArgumentParser(description='Evaluate MOSS reports.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to show TP/FP/TN/FN pairs.')
    parser.add_argument('--debug-threshold', type=int, default=25, help='Threshold to use for debug mode.')
    args = parser.parse_args()

    report_dir = 'moss_reports'
    results = {}

    print(f"Parsing MOSS reports from '{report_dir}'...")
    for filename in os.listdir(report_dir):
        if filename.endswith('_report.html'):
            match = re.match(r'(\d+)_report\.html', filename)
            if match:
                index = int(match.group(1))
                file_path = os.path.join(report_dir, filename)
                similarity = parse_moss_report(file_path)
                if similarity is not None:
                    results[index] = similarity
    print(f"Parsed {len(results)} reports.")

    sim_matches = {k: v for k, v in results.items() if 0 <= k <= 1999}
    not_sim_matches = {k: v for k, v in results.items() if 2000 <= k <= 3999}
    
    if args.debug:
        debug_evaluation(sim_matches, not_sim_matches, args.debug_threshold)
        return # Exit after debugging

    total_sim = len(sim_matches)
    total_not_sim = len(not_sim_matches)

    if total_sim == 0 and total_not_sim == 0:
        print("No matches found. Exiting.")
        return

    output_file = 'moss_evaluation_results.csv'
    print(f"Calculating metrics for thresholds and saving to '{output_file}'...")

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            'threshold', 
            'tp', 'fn', 'fp', 'tn',
            'tpr', 'fnr', 'fpr', 'tnr', 
            'accuracy', 'precision', 'f1_score'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for threshold in range(0, 101, 5):
            tp = sum(1 for v in sim_matches.values() if v > threshold)
            fn = total_sim - tp
            
            fp = sum(1 for v in not_sim_matches.values() if v > threshold)
            tn = total_not_sim - fp
            
            tpr = tp / total_sim if total_sim > 0 else 0
            fnr = fn / total_sim if total_sim > 0 else 0
            fpr = fp / total_not_sim if total_not_sim > 0 else 0
            tnr = tn / total_not_sim if total_not_sim > 0 else 0
            
            accuracy = (tp + tn) / (total_sim + total_not_sim) if (total_sim + total_not_sim) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1_score = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0

            writer.writerow({
                'threshold': threshold,
                'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
                'tpr': f'{tpr:.4f}', 'fnr': f'{fnr:.4f}', 
                'fpr': f'{fpr:.4f}', 'tnr': f'{tnr:.4f}',
                'accuracy': f'{accuracy:.4f}',
                'precision': f'{precision:.4f}',
                'f1_score': f'{f1_score:.4f}'
            })

    print("--- MOSS Evaluation Complete ---")
    print(f"Results saved to {output_file}")
    
    if sim_matches:
        avg_sim = np.mean(list(sim_matches.values()))
        print(f"\nAverage similarity for 'similar' pairs: {avg_sim:.2f}%")

    if not_sim_matches:
        avg_not_sim = np.mean(list(not_sim_matches.values()))
        print(f"Average similarity for 'not similar' pairs: {avg_not_sim:.2f}%")

if __name__ == '__main__':
    main()