import os
import json
from absl import app
from absl import flags
import modules.rcomplexity as rcomplexity
import fcntl
import math

dataset = {}
outcome = {}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "problem_id", "", help="If specified, only problem with ID will be compared"
)
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

flags.DEFINE_bool("testing_mode", False, help="Whether to enable testing")


def read_all_json_files_recursive(root_path, dataset):
    if not FLAGS.testing_mode and os.path.exists(
        "/Users/raresraf/code/project-martial/samples/rcomplexity/train_rcomplexity.json"
    ):
        with open(
            "/Users/raresraf/code/project-martial/samples/rcomplexity/train_rcomplexity.json",
            "r",
        ) as fp:
            dataset.update(json.load(fp))
        return
    for root, _, files in os.walk(root_path):
        for file in files:
            if file.endswith("PROCESSED.RAF"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    data = json.load(f)

                    problem = os.path.basename(os.path.dirname(file_path))
                    if "_" in problem:
                        continue
                    if not dataset.get(problem, None):
                        dataset[problem] = []
                    dataset[problem].append(data)


def load_common_labels():
    with open(
        "/Users/raresraf/code/TheInputsCodeforces/metadata/metadata.json"
    ) as json_file:
        labels = json.load(json_file)
    all_problems = [
        "281A",
        "263A",
        "118A",
        "158A",
        "160A",
        "69A",
        "58A",
        "546A",
        "670A",
        "266B",
        "339A",
        "110A",
        "266A",
        "96A",
        "112A",
        "236A",
        "231A",
        "116A",
        "50A",
        "282A",
        "71A",
        "898B",
        "1221C",
        "1343C",
        "1140D",
        "753A",
        "1366A",
        "1426C",
        "50C",
        "64A",
        "1426B",
        "50B",
    ]
    common_labels = {}
    for p1 in all_problems:
        for p2 in all_problems:
            has_common_label = False

            if not common_labels.get(p1, None):
                common_labels[p1] = {}
            if not common_labels.get(p2, None):
                common_labels[p2] = {}
            if not labels.get(p1, None):
                common_labels[p1][p2] = False
                common_labels[p2][p1] = False
                continue
            if not labels.get(p2, None):
                common_labels[p1][p2] = False
                common_labels[p2][p1] = False
                continue
            for pp1 in labels[p1]:
                for pp2 in labels[p2]:
                    if pp1 == pp2:
                        has_common_label = True

            if has_common_label:
                common_labels[p1][p2] = True
                common_labels[p2][p1] = True
            else:
                common_labels[p1][p2] = False
                common_labels[p2][p1] = False

    return common_labels


def main(_):
    common_labels = load_common_labels()

    root_directory_path = (
        "/Users/raresraf/code/TheOutputsCodeforces/splitted/train/atomic_perf/"
    )
    if FLAGS.testing_mode:
        root_directory_path = (
            "/Users/raresraf/code/TheOutputsCodeforces/splitted/test/atomic_perf/"
        )

    read_all_json_files_recursive(root_directory_path, dataset)

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

    play_game_current_points = 0
    play_game_total_points = 0
    loss = 0

    for k1 in dataset.keys():
        if FLAGS.problem_id != "" and k1 != FLAGS.problem_id:
            continue
        for k2 in dataset.keys():
            for f1 in dataset[k1]:
                if not f1.get("path", None):
                    continue
                for f2 in dataset[k2]:
                    if not f2.get("path", None):
                        continue
                    rca.fileJSON["file1"] = f1
                    rca.fileJSON["file2"] = f2
                    _, _, similarity, _, _, _, _ = (
                        rca.find_complexity_similarity_with_as()
                    )

                    want_similarity = 0
                    scale = 3
                    if k1 == k2:
                        want_similarity = 1
                    if common_labels[k1][k2]:
                        want_similarity = 1
                        scale = 1

                    loss += scale * (
                        -want_similarity * infzerolog(similarity)
                        - (1 - want_similarity) * infzerolog(1 - similarity)
                    )
                    play_game_current_points += (
                        scale * 1
                        if want_similarity == int(similarity >= FLAGS.threshold)
                        else 0
                    )
                    play_game_total_points += scale * 1

                    # print(f"{similarity}, {k1}, {k2}\n")
                    # out_file.write(f"{similarity}, {k1}, {k2}\n")

    msg = {
        "problem_id": FLAGS.problem_id,
        "threshold": FLAGS.threshold,
        "loss": loss,
        "X": rca.X,
        "play_game_current_points": play_game_current_points,
        "play_game_total_points": play_game_total_points,
        "accuracy": play_game_current_points / play_game_total_points,
    }
    # print(msg)

    if FLAGS.testing_mode:
        with open(
            f"/Users/raresraf/code/project-martial/samples/rcomplexity/test_rcomplexity_dataset_results.json",
            "a",
        ) as fp:
            fcntl.flock(fp, fcntl.LOCK_EX)
            json.dump(msg, fp)
            fp.write("\n")
            fcntl.flock(fp, fcntl.LOCK_UN)
    else:
        with open(
            f"/Users/raresraf/code/project-martial/samples/rcomplexity/rcomplexity_dataset_results.json",
            "a",
        ) as fp:
            fcntl.flock(fp, fcntl.LOCK_EX)
            json.dump(msg, fp)
            fp.write("\n")
            fcntl.flock(fp, fcntl.LOCK_UN)


def infzerolog(x):
    if x < 3.720076e-44:
        return -100
    return math.log(x)


if __name__ == "__main__":
    app.run(main)
