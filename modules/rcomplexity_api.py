import modules.rcomplexity as rcomplexity
import modules.utils as utils
from modules.rcomplexity import RComplexityAnalysis


def run(upload_dict):
    resp = {}
    rca = common_run(upload_dict)
    
    lines_in_1, lines_in_2 = rca.find_critical_matches()
    resp.update(feed_matches("critical", lines_in_1, lines_in_2))

    lines_in_1, lines_in_2 = rca.find_complexity_similarity()
    resp.update(feed_matches("complexity", lines_in_1, lines_in_2))
    return resp

def feed_matches(class_of_issue, lines_in_1, lines_in_2):
    resp = {class_of_issue: []}
    for count, item in enumerate(lines_in_1):
        resp[class_of_issue].append({"file1": lines_in_1[count], "file2": lines_in_2[count]})
    return resp

def common_run(upload_dict) -> RComplexityAnalysis:
    rca = RComplexityAnalysis()
    rca.link_to_token(utils.generate_token())

    if upload_dict.get("file1", None):
        rca.load_birthmark("file1", upload_dict["file1"])
    if upload_dict.get("file2", None):
        rca.load_birthmark("file2", upload_dict["file2"])
    return rca
