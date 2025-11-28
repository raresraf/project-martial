from flask import request, jsonify
import modules.network_traffic_analysis as nta_module

nta_analyzer = nta_module.NetworkTrafficAnalysis()

def run(upload_dict):
    resp = {}
    
    file1_content = upload_dict.get("file1")
    file2_content = upload_dict.get("file2")

    ngram_param = request.args.get('ngram', type=int)
    n_gram_selection = ngram_param if ngram_param in [2, 3, 4] else 4
    print(f"DEBUG: n_gram_selection = {n_gram_selection}")

    nta_analyzer.load_file_content("file1", file1_content)
    nta_analyzer.load_file_content("file2", file2_content)

    analysis_results = nta_analyzer.compare_network_traffic("file1", "file2", n_gram_selection)
    resp.update(analysis_results)
    
    return resp