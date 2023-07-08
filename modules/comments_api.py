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

def comments_common(ca):
    report = {}

    common_list, lines_in_1, lines_in_2 = ca.analyze_2_files()
    feed_common_list_in_report("comment_exact_lines_files", report, ca.token, common_list, lines_in_1, lines_in_2)

    common_list, lines_in_1, lines_in_2 = ca.analyze_2_files_fuzzy()
    feed_common_list_in_report("comment_fuzzy_lines_files", report, ca.token, common_list, lines_in_1, lines_in_2)

    common_list, lines_in_1, lines_in_2 = ca.analyze_2_files_spacy_core_web()
    feed_common_list_in_report("comment_spacy_core_web_lines_files", report, ca.token, common_list, lines_in_1, lines_in_2)

    report_without_overlap = remove_comments_superset(report)
    return report_without_overlap


def feed_common_list_in_report(entry, report, token, common_list, lines_in_1, lines_in_2):
    report[entry] = []
    print(
        f"[traceID: {token}] common_list for {entry}: found {len(common_list)} common sequences")
    for idx, _ in enumerate(common_list):
        report[entry].append(
            {
                "file1": lines_in_1[idx],
                "file2": lines_in_2[idx],
            },
        )
        print(
            f"[traceID: {token}] Finished analyzing {entry}: {idx}/{len(common_list)}")


def remove_comments_superset(report: dict) -> dict:
    """
    [
        {
            "file1": [11],
            "file2": [15, 16, 18],
        },
        {
            "file1": [11],
            "file2": [16, 18],
        },
        {
            "file1": [11],
            "file2": [18],
        },
    ]

    Probably only 
        {
            "file1": [11],
            "file2": [18],
        },
    is interesting.
    """
    report_without_overlap = {}
    for e_name, e_vals in report.items():
        report_without_overlap[e_name] = []
        for e_val in e_vals:
            is_superset = False
            for e_val_against in e_vals:
                if (len(e_val_against["file1"]) <= len(e_val["file1"])) and (len(e_val_against["file2"]) <= len(e_val["file2"])) and (len(e_val_against["file1"]) + len(e_val_against["file2"]) < len(e_val["file1"]) + len(e_val["file2"])):
                    if all(f1 in e_val["file1"] for f1 in e_val_against["file1"]) and all(f2 in e_val["file2"] for f2 in e_val_against["file2"]):
                        is_superset = True
                        break
            if not is_superset:
                report_without_overlap[e_name].append(e_val)


    return report_without_overlap



def generate_token():
    token = secrets.token_hex(7)
    print(f"Received custom request with token {token}")
    return token