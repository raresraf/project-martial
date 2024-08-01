import os
import json
from absl import app
from absl import flags
from sklearn.metrics import recall_score, precision_score, accuracy_score
import modules.rcomplexity as rcomplexity
import modules.drivers.rcomplexity_driver as rcomplexity_driver
import fcntl
import math
from tqdm import tqdm
from threading import Thread, Lock
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import random
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

train_dataset = {}
test_dataset = {}
outcome = {}


flags.DEFINE_float("lr", "0.000001", help="learning rate")
flags.DEFINE_integer("batch_size", "32", help="batch size")
flags.DEFINE_integer("total_iter", "100", help="total number of iterations")
flags.DEFINE_bool(
    "skip_similar_but_not_identic_problems",
    True,
    help="skip_similar_but_not_identic_problems",
)
flags.DEFINE_bool(
    "run_balanced_tests",
    True,
    help="If true, the test dataset will be split 50%%-50%%",
)

FLAGS = flags.FLAGS


def main(_):
    common_labels = rcomplexity_driver.load_common_labels()
    root_train_directory_path = (
        "/Users/raresraf/code/TheOutputsCodeforces/splitted/train/atomic_perf/"
    )
    rcomplexity_driver.read_all_json_files_recursive(
        root_train_directory_path, train_dataset
    )
    root_test_directory_path = (
        "/Users/raresraf/code/TheOutputsCodeforces/splitted/test/atomic_perf/"
    )
    rcomplexity_driver.read_all_json_files_recursive(
        root_test_directory_path, test_dataset
    )

    print(len(test_dataset))

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

    lr = FLAGS.lr
    total_iter = FLAGS.total_iter
    batch_size = FLAGS.batch_size
    curr_batch = 0

    total_ops = 0
    for k1 in train_dataset.keys():
        for k2 in train_dataset.keys():
            total_ops += len(train_dataset[k1]) * len(train_dataset[k2])

    similar = 0
    not_similar = 0
    all_train_k1_k2_f1_f2_pairs = []
    if lr != 0:
        for k1 in train_dataset.keys():
            for k2 in train_dataset.keys():
                for f1 in train_dataset[k1]:
                    if not f1.get("path", None):
                        continue
                    for f2 in train_dataset[k2]:
                        if not f2.get("path", None):
                            continue
                        if f1.get("path", None) == f2.get("path"):
                            continue
                        if (
                            FLAGS.skip_similar_but_not_identic_problems
                            and common_labels[k1][k2]
                            and k1 != k2
                        ):
                            continue

                        all_train_k1_k2_f1_f2_pairs.append((k1, k2, f1, f2))
                        if k1 == k2 or common_labels[k1][k2]:
                            similar += 1
                        else:
                            not_similar += 1

    min_similar_not_similar = similar if similar < not_similar else not_similar

    similar = 0
    not_similar = 0
    selected_train_k1_k2_f1_f2_pairs = []
    for k1, k2, f1, f2 in all_train_k1_k2_f1_f2_pairs:
        if k1 == k2 or common_labels[k1][k2]:
            if similar > min_similar_not_similar:
                continue
            else:
                similar += 1
                selected_train_k1_k2_f1_f2_pairs.append((k1, k2, f1, f2))
        else:
            if not_similar > min_similar_not_similar:
                continue
            else:
                not_similar += 1
                selected_train_k1_k2_f1_f2_pairs.append((k1, k2, f1, f2))

    print(f"similar (train) = {similar}, not_similar (train) = {not_similar}")

    selected_train_k1_k2_f1_f2_pairs = tuple(
        random.sample(
            selected_train_k1_k2_f1_f2_pairs, len(selected_train_k1_k2_f1_f2_pairs)
        )
    )

    all_test_k1_k2_f1_f2_pairs = []
    for k1 in test_dataset.keys():
        for k2 in test_dataset.keys():
            for f1 in test_dataset[k1]:
                if not f1.get("path", None):
                    continue
                for f2 in test_dataset[k2]:
                    if not f2.get("path", None):
                        continue
                    if (
                        FLAGS.skip_similar_but_not_identic_problems
                        and common_labels[k1][k2]
                        and k1 != k2
                    ):
                        continue

                    all_test_k1_k2_f1_f2_pairs.append((k1, k2, f1, f2))
    print(f"similar (test) = {similar}, not_similar (test) = {not_similar}")

    similar = 0
    not_similar = 0
    selected_test_k1_k2_f1_f2_pairs = []
    for k1, k2, f1, f2 in all_test_k1_k2_f1_f2_pairs:
        if k1 == k2 or common_labels[k1][k2]:
            if similar > min_similar_not_similar and FLAGS.run_balanced_tests:
                continue
            similar += 1
        else:
            if not_similar > min_similar_not_similar and FLAGS.run_balanced_tests:
                continue
            not_similar += 1
        selected_test_k1_k2_f1_f2_pairs.append((k1, k2, f1, f2))

    with tqdm(
        total=total_iter * len(selected_train_k1_k2_f1_f2_pairs)
        + total_iter * len(selected_test_k1_k2_f1_f2_pairs),
        desc="Processing",
    ) as pbar:
        for i in range(total_iter):
            loss = 0
            y_pred = []
            y_true = []
            big_dc = [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]

            similar = 0
            not_similar = 0
            for k1, k2, f1, f2 in selected_train_k1_k2_f1_f2_pairs:
                local_rca = rcomplexity.RComplexityAnalysis()
                local_rca.disable_find_line = True
                local_rca.X = rca.X
                local_rca.fileJSON["file1"] = f1
                local_rca.fileJSON["file2"] = f2
                _, _, similarity, a1s, a2s, a3s, a4s = (
                    local_rca.find_complexity_similarity_with_as()
                )
                aijs = [
                    [a1s[0], a2s[0], a3s[0], a4s[0]],
                    [a1s[1], a2s[1], a3s[1], a4s[1]],
                    [a1s[2], a2s[2], a3s[2], a4s[2]],
                    [a1s[3], a2s[3], a3s[3], a4s[3]],
                    [a1s[4], a2s[4], a3s[4], a4s[4]],
                    [a1s[5], a2s[5], a3s[5], a4s[5]],
                    [a1s[6], a2s[6], a3s[6], a4s[6]],
                    [a1s[7], a2s[7], a3s[7], a4s[7]],
                    [a1s[8], a2s[8], a3s[8], a4s[8]],
                ]

                want_similarity = 0
                if k1 == k2:
                    want_similarity = 1
                if common_labels[k1][k2]:
                    want_similarity = 1

                loss += -want_similarity * rcomplexity_driver.infzerolog(similarity) - (
                    1 - want_similarity
                ) * rcomplexity_driver.infzerolog(1 - similarity)

                y_pred.append(int(similarity >= FLAGS.threshold))
                y_true.append(want_similarity)

                dcoef = -divifzero(want_similarity, similarity) + divifzero(
                    1 - want_similarity, 1 - similarity
                )

                sac = sum_of_all_c(local_rca)
                sac_square = sac**2
                sacaij = sum_of_all_c_scaled_by_as(local_rca, aijs)

                for i in range(len(local_rca.X)):
                    for j in range(len(local_rca.X[i])):
                        big_dc[i][j] += dcoef * (aijs[i][j] * sac - sacaij) / sac_square

                curr_batch += 1
                if curr_batch == batch_size:
                    run_batch_update(rca, lr, big_dc)
                    curr_batch = 0
                    pbar.update(batch_size)

            # Ignore last batch by commenting below line:
            # run_batch_update(rca, lr, big_dc)
            curr_batch = 0
            pbar.update(curr_batch)

            # === Start of EVAL CODE ===
            loss_test = 0
            y_pred_test = []
            y_true_test = []
            for k1, k2, f1, f2 in selected_test_k1_k2_f1_f2_pairs:
                local_rca = rcomplexity.RComplexityAnalysis()
                local_rca.disable_find_line = True
                local_rca.X = rca.X
                local_rca.fileJSON["file1"] = f1
                local_rca.fileJSON["file2"] = f2
                _, _, similarity, _, _, _, _ = (
                    local_rca.find_complexity_similarity_with_as()
                )

                want_similarity = 0
                if k1 == k2:
                    want_similarity = 1
                if common_labels[k1][k2]:
                    want_similarity = 1

                loss_test += -want_similarity * rcomplexity_driver.infzerolog(
                    similarity
                ) - (1 - want_similarity) * rcomplexity_driver.infzerolog(
                    1 - similarity
                )
                y_pred_test.append(int(similarity >= FLAGS.threshold))
                y_true_test.append(want_similarity)
                pbar.update(1)
            # === End of EVAL CODE ===

            print(classification_report(y_true_test, y_pred_test))
            print(confusion_matrix(y_true_test, y_pred_test))
            msg = {
                "train_accuracy": accuracy_score(y_true, y_pred),
                "train_recall": recall_score(y_true, y_pred),
                "train_precision": precision_score(y_true, y_pred),
                "train_loss": loss,
                "test_accuracy": accuracy_score(y_true_test, y_pred_test),
                "test_recall": recall_score(y_true_test, y_pred_test),
                "test_precision": precision_score(y_true_test, y_pred_test),
                "test_loss": loss_test,
                "X": rca.X,
            }
            with open(
                f"/Users/raresraf/code/project-martial/samples/rcomplexity/rcomplexity_train_results_{FLAGS.lr}_{FLAGS.batch_size}.json",
                "a",
            ) as fp:
                json.dump(msg, fp)
                fp.write("\n")


def run_batch_update(rca, lr, big_dc):
    for i in range(len(rca.X)):
        for j in range(len(rca.X[i])):
            if rca.X[i][j] < lr * big_dc[i][j]:
                rca.X[i][j] = 0
            else:
                rca.X[i][j] = rca.X[i][j] - lr * big_dc[i][j]

    max_cij = 0
    for i in range(len(rca.X)):
        for j in range(len(rca.X[i])):
            if rca.X[i][j] > max_cij:
                max_cij = rca.X[i][j]
    for i in range(len(rca.X)):
        for j in range(len(rca.X[i])):
            rca.X[i][j] = rca.X[i][j] / max_cij

    big_dc = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]


def sum_of_all_c(rca):
    sum = 0
    for l in rca.X:
        for c in l:
            sum += c
    return sum


def sum_of_all_c_scaled_by_as(rca, aijs):
    sum = 0
    for i in range(len(rca.X)):
        for j in range(len(rca.X[i])):
            sum += rca.X[i][j] * aijs[i][j]
    return sum


def divifzero(x, y):
    if x == 0:
        return 0
    if y < 0.1:
        return x / 0.1
    return x / y


if __name__ == "__main__":
    app.run(main)
