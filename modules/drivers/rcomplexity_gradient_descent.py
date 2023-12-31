import os
import json
from absl import app
from absl import flags
import modules.rcomplexity as rcomplexity
import modules.drivers.rcomplexity_driver as rcomplexity_driver
import fcntl
import math
from tqdm import tqdm
from threading import Thread, Lock
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED


mutex = Lock()
dataset = {}
outcome = {}
global_loss = {"loss": 0}

FLAGS = flags.FLAGS


def main(_):
    common_labels = rcomplexity_driver.load_common_labels()
    root_directory_path = '/Users/raresraf/code/TheOutputsCodeforces/splitted/train/atomic_perf/'
    rcomplexity_driver.read_all_json_files_recursive(
        root_directory_path, dataset)

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

    lr = 0.00001
    total_iter = 10
    futures = set()

    total_op = total_iter
    for k1 in dataset.keys():
        for k2 in dataset.keys():
            total_op += len(dataset[k1]) * len(dataset[k2])


    with tqdm(total=total_iter * total_op, desc="Processing") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for i in range(total_iter):

                global_loss["loss"] = 0
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
                for k1 in dataset.keys():
                    for k2 in dataset.keys():
                        for f1 in dataset[k1]:
                            if not f1.get("path", None):
                                pbar.update(1)
                                continue
                            for f2 in dataset[k2]:
                                if not f2.get("path", None):
                                    pbar.update(1)
                                    continue

                                local_rca = rcomplexity.RComplexityAnalysis()
                                local_rca.disable_find_line = True
                                local_rca.X = rca.X
                                local_rca.fileJSON["file1"] = f1
                                local_rca.fileJSON["file2"] = f2
                                futures.add(executor.submit(run_gradient, k1, k2, common_labels,
                                                            local_rca, big_dc))
                                pbar.update(1)

                completed, futures = wait(futures, return_when=ALL_COMPLETED)
                for i in range(len(rca.X)):
                    for j in range(len(rca.X[i])):
                        rca.X[i][j] = -lr * big_dc[i][j]

                msg = {
                    "loss": global_loss["loss"],
                    "X": rca.X,
                }
                with open(f"/Users/raresraf/code/project-martial/samples/rcomplexity/rcomplexity_train_results.json", 'a') as fp:
                    fcntl.flock(fp, fcntl.LOCK_EX)
                    json.dump(msg, fp)
                    fp.write("\n")
                    fcntl.flock(fp, fcntl.LOCK_UN)
                


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
    if y < 0.002478:
        return 6
    return x/y


def run_gradient(k1, k2, common_labels, local_rca, big_dc):
    _, _, similarity, a1s, a2s, a3s, a4s = local_rca.find_complexity_similarity_with_as()
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
    dc = [
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

    want_similarity = 0
    scale = 3
    if k1 == k2:
        want_similarity = 1
    if common_labels[k1][k2]:
        want_similarity = 1
        scale = 1

    loss = scale * (-want_similarity * rcomplexity_driver.infzerolog(similarity) -
                    (1-want_similarity) * rcomplexity_driver.infzerolog(1-similarity))

    dcoef = -divifzero(want_similarity, similarity) + \
        divifzero(1-want_similarity, 1-similarity)

    sac = sum_of_all_c(local_rca)
    sac_square = sac ** 2
    sacaij = sum_of_all_c_scaled_by_as(local_rca, aijs)

    for i in range(len(local_rca.X)):
        for j in range(len(local_rca.X[i])):
            dc[i][j] = dcoef * (aijs[i][j] * sac - sacaij) / sac_square

    with mutex:
        for i in range(len(local_rca.X)):
            for j in range(len(local_rca.X[i])):
                big_dc[i][j] += dc[i][j]
                global_loss["loss"] += loss


if __name__ == '__main__':
    app.run(main)
