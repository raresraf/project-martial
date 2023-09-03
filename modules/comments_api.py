import secrets

from modules.comments import CommentsAnalysis
import modules.comments as comments
import modules.comments_helpers as comments_helpers


def run(upload_dict):
    ca = common_run(upload_dict)
    return comments_common(ca)


def common_run(upload_dict) -> CommentsAnalysis:
    ca = CommentsAnalysis()
    ca.link_to_token(generate_token())

    if upload_dict.get("file1", None):
        ca.load_text("file1", upload_dict["file1"])
    if upload_dict.get("file2", None):
        ca.load_text("file2", upload_dict["file2"])
    return ca


def run_text(upload_dict):
    ca = common_run(upload_dict)
    return comments_common(ca, feed_method=feed_common_list_in_text_report)


def feed_common_list_in_report(entry, report, token, common_list, lines_in_1, lines_in_2, keep_text=False):
    report[entry] = []
    list_len = len(common_list)
    print_report = [list_len / 10, 2 * list_len / 10, 3 * list_len / 10, 4 * list_len / 10, 5 * list_len /
                    10, 6 * list_len / 10, 7 * list_len / 10, 8 * list_len / 10, 9 * list_len / 10, list_len - 1]
    print(
        f"[traceID: {token}] common_list for {entry}: found {list_len} common sequences")
    for idx, _ in enumerate(common_list):
        update = {
            "file1": lines_in_1[idx],
            "file2": lines_in_2[idx],
        }
        if keep_text:
            update["file1_text"] = str(common_list[idx][0])
            update["file2_text"] = str(common_list[idx][1])

        report[entry].append(update)
        if idx in print_report:
            print(
                f"[traceID: {token}] Finished analyzing {entry}: {idx}/{list_len - 1}", end='\r')
    print("")


def feed_common_list_in_text_report(entry, report, token, common_list, lines_in_1, lines_in_2):
    feed_common_list_in_report(
        entry, report, token, common_list, lines_in_1, lines_in_2, keep_text=True)


def comments_common(ca: CommentsAnalysis, feed_method=feed_common_list_in_report):
    report = {}

    common_list_comment_exact_lines_files, lines_in_1, lines_in_2 = ca.analyze_2_files(
        "Exact match", ca.exact_match_similarity, ca.comm_to_seq_1)
    feed_method("comment_exact_lines_files",
                report, ca.token, common_list_comment_exact_lines_files, lines_in_1, lines_in_2)

    common_list_comment_fuzzy_lines_files, lines_in_1, lines_in_2 = ca.analyze_2_files(
        "Fuzzy match", ca.fuzzy_similarity, ca.comm_to_seq_6)
    feed_method("comment_fuzzy_lines_files",
                report, ca.token, common_list_comment_fuzzy_lines_files, lines_in_1, lines_in_2)

    common_list_comment_spacy_core_web_lines_files, lines_in_1, lines_in_2 = ca.analyze_2_files(
        "SpacyCoreWeb match", ca.spacy_similarity, ca.comm_to_seq_doc)
    feed_method("comment_spacy_core_web_lines_files",
                report, ca.token, common_list_comment_spacy_core_web_lines_files, lines_in_1, lines_in_2)

    if ca.enable_elmo:
        common_list_comment_elmo_lines_files, lines_in_1, lines_in_2 = ca.analyze_2_files(
            "Elmo match", ca.elmo_similarity, ca.comm_to_seq_elmo, True)
        feed_method(
            "comment_elmo_lines_files", report, ca.token, common_list_comment_elmo_lines_files, lines_in_1, lines_in_2)

    report_without_overlap = remove_comments_superset_that_were_not_deduped_before(
        report)

    return report_without_overlap


def remove_comments_superset_that_were_not_deduped_before(report: dict) -> dict:
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
        for e_idx, e_val in enumerate(e_vals):
            print(f"progress {e_idx}/{len(e_vals)}", end="\r")
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
