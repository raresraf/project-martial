import secrets

from modules.comments import CommentsAnalysis


def run(upload_dict):
    ca = CommentsAnalysis()
    ca.link_to_token(generate_token())

    if upload_dict.get("file1", None):
        ca.load_text("file1", upload_dict["file1"])
    if upload_dict.get("file2", None):
        ca.load_text("file2", upload_dict["file2"])
    return comments_common(ca)


def run_custom(custom_1, custom_2):
    ca = CommentsAnalysis()
    ca.link_to_token(generate_token())

    with open(custom_1, 'r') as f:
        upload_dict["file1"] = f.read()
        ca.load_text("file1", upload_dict["file1"])
    with open(custom_2, 'r') as f:
        upload_dict["file2"] = f.read()
        ca.load_text("file2", upload_dict["file2"])
    return comments_common(ca)

def comments_common(ca):
    report = {
        "comment_exact_lines_files": [],
        "comment_fuzzy_lines_files": [],
        "comment_spacy_core_web_lines_files": [],
    }

    common_list, lines_in_1, lines_in_2 = ca.analyze_2_files()
    print(
        f"[traceID: {ca.token}] common_list for analyze_2_files: found {len(common_list)} common sequences")
    for idx, _ in enumerate(common_list):
        report["comment_exact_lines_files"].append(
            {
                "file1": lines_in_1[idx],
                "file2": lines_in_2[idx],
            },
        )
        print(
            f"[traceID: {ca.token}] Finished analyzing comment_exact_lines_files: {idx}/{len(common_list)}")

    common_list, lines_in_1, lines_in_2 = ca.analyze_2_files_fuzzy()
    print(
        f"[traceID: {ca.token}] common_list for analyze_2_files_fuzzy: found {len(common_list)} common sequences")
    for idx, _ in enumerate(common_list):
        report["comment_fuzzy_lines_files"].append(
            {
                "file1": lines_in_1[idx],
                "file2": lines_in_2[idx],
            },
        )
        print(
            f"[traceID: {ca.token}] Finished analyzing comment_fuzzy_lines_files: {idx}/{len(common_list)}")

    common_list, lines_in_1, lines_in_2 = ca.analyze_2_files_spacy_core_web()
    print(
        f"[traceID: {ca.token}] common_list for analyze_2_files_spacy_core_web: found {len(common_list)} common sequences")
    for idx, _ in enumerate(common_list):
        report["comment_spacy_core_web_lines_files"].append(
            {
                "file1": lines_in_1[idx],
                "file2": lines_in_2[idx],
            },
        )
        print(
            f"[traceID: {ca.token}] Finished analyzing comment_spacy_core_web_lines_files: {idx}/{len(common_list)}")

    return report


def generate_token():
    token = secrets.token_hex(7)
    print(f"Received custom request with token {token}")
    return token