
import subprocess
import concurrent.futures
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import itertools


def run_bazel_command(target, extra_args):
    bazel_command = [
        'bazel',
        'run',
        target,
        '--',
    ]
    bazel_command.extend(extra_args)
    subprocess.run(bazel_command, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


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

# Iteration 1
possible_th_ranges = [x / 50 for x in range(51)]
all_possible_f1_ranges = [1]
all_possible_f2_ranges = [0.66]
all_possible_f3_ranges = [0.5]
all_possible_f4_ranges = [0.01]

# # Iteration 2
# possible_th_ranges = [0.33]
# all_possible_f1_ranges = [1, 0]
# all_possible_f2_ranges = [0.66]
# all_possible_f3_ranges = [0.5]
# all_possible_f4_ranges = [0.01]
# 
# # Iteration 3
# possible_th_ranges = [0.1, 0.25, 0.33]
# all_possible_f1_ranges = [0.75, 0.5, 0.25]
# all_possible_f2_ranges = [0.66]
# all_possible_f3_ranges = [0.5]
# all_possible_f4_ranges = [0.01]

all_possible_c_ranges = [
    all_possible_f1_ranges, all_possible_f2_ranges, all_possible_f3_ranges, all_possible_f4_ranges, all_possible_f1_ranges, all_possible_f2_ranges, all_possible_f3_ranges, all_possible_f4_ranges, all_possible_f1_ranges, all_possible_f2_ranges, all_possible_f3_ranges, all_possible_f4_ranges, all_possible_f1_ranges, all_possible_f2_ranges, all_possible_f3_ranges, all_possible_f4_ranges, all_possible_f1_ranges, all_possible_f2_ranges,
    all_possible_f3_ranges, all_possible_f4_ranges, all_possible_f1_ranges, all_possible_f2_ranges, all_possible_f3_ranges, all_possible_f4_ranges, all_possible_f1_ranges, all_possible_f2_ranges, all_possible_f3_ranges, all_possible_f4_ranges, all_possible_f1_ranges, all_possible_f2_ranges, all_possible_f3_ranges, all_possible_f4_ranges, all_possible_f1_ranges, all_possible_f2_ranges, all_possible_f3_ranges, all_possible_f4_ranges,
]
target = 'modules/drivers:rcomplexity_driver'


total = len(possible_th_ranges) * len(all_problems) * len(all_possible_f1_ranges) ** (9) * \
    len(all_possible_f2_ranges) ** (9) * \
    len(all_possible_f3_ranges) ** (9) * len(all_possible_f4_ranges) ** (9)
futures = set()

with tqdm(total=total, desc="Processing") as pbar:
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        for c11, c12, c13, c14, c21, c22, c23, c24, c31, c32, c33, c34, c41, c42, c43, c44, c51, c52, c53, c54, c61, c62, c63, c64, c71, c72, c73, c74, c81, c82, c83, c84, c91, c92, c93, c94 in itertools.product(*all_possible_c_ranges):
            for threshold in possible_th_ranges:
                for problem in all_problems:
                    args = [
                        '--problem_id=' + problem,
                        '--threshold=' + str(threshold),
                        '--c11=' + str(c11),
                        '--c12=' + str(c12),
                        '--c13=' + str(c13),
                        '--c14=' + str(c14),
                        '--c21=' + str(c21),
                        '--c22=' + str(c22),
                        '--c23=' + str(c23),
                        '--c24=' + str(c24),
                        '--c31=' + str(c31),
                        '--c32=' + str(c32),
                        '--c33=' + str(c33),
                        '--c34=' + str(c34),
                        '--c41=' + str(c41),
                        '--c42=' + str(c42),
                        '--c43=' + str(c43),
                        '--c44=' + str(c44),
                        '--c51=' + str(c51),
                        '--c52=' + str(c52),
                        '--c53=' + str(c53),
                        '--c54=' + str(c54),
                        '--c61=' + str(c61),
                        '--c62=' + str(c62),
                        '--c63=' + str(c63),
                        '--c64=' + str(c64),
                        '--c71=' + str(c71),
                        '--c72=' + str(c72),
                        '--c73=' + str(c73),
                        '--c74=' + str(c74),
                        '--c81=' + str(c81),
                        '--c82=' + str(c82),
                        '--c83=' + str(c83),
                        '--c84=' + str(c84),
                        '--c91=' + str(c91),
                        '--c92=' + str(c92),
                        '--c93=' + str(c93),
                        '--c94=' + str(c94),
                    ]

                    if len(futures) >= 1000:
                        completed, futures = wait(
                            futures, return_when=FIRST_COMPLETED)

                    futures.add(executor.submit(
                        run_bazel_command, target, args))
                    pbar.update(1)
