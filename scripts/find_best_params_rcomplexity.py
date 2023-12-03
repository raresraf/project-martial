import json
import subprocess

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


def run_test_bazel_command(target, threshold, c11, c12, c13, c14, c21, c22, c23, c24, c31, c32, c33, c34, c41, c42, c43, c44, c51, c52, c53, c54, c61, c62, c63, c64, c71, c72, c73, c74, c81, c82, c83, c84, c91, c92, c93, c94):
    for problem in all_problems:
        extra_args = ([
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
            '--testing_mode=true',
        ])

        bazel_command = [
            'bazel',
            'run',
            target,
            '--',
        ]
        bazel_command.extend(extra_args)
        subprocess.run(bazel_command, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)


def run_on_file_path(file_path):
    results = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())

            threshold = data["threshold"]
            X = (data["X"][0][0],
                 data["X"][0][1],
                 data["X"][0][2],
                 data["X"][0][3],
                 data["X"][1][0],
                 data["X"][1][1],
                 data["X"][1][2],
                 data["X"][1][3],
                 data["X"][2][0],
                 data["X"][2][1],
                 data["X"][2][2],
                 data["X"][2][3],
                 data["X"][3][0],
                 data["X"][3][1],
                 data["X"][3][2],
                 data["X"][3][3],
                 data["X"][4][0],
                 data["X"][4][1],
                 data["X"][4][2],
                 data["X"][4][3],
                 data["X"][5][0],
                 data["X"][5][1],
                 data["X"][5][2],
                 data["X"][5][3],
                 data["X"][6][0],
                 data["X"][6][1],
                 data["X"][6][2],
                 data["X"][6][3],
                 data["X"][7][0],
                 data["X"][7][1],
                 data["X"][7][2],
                 data["X"][7][3],
                 data["X"][8][0],
                 data["X"][8][1],
                 data["X"][8][2],
                 data["X"][8][3],)

            key = (threshold, X)
            if not results.get(key, None):
                results[key] = {
                    'play_game_current_points': 0,
                    'play_game_total_points': 0,
                    'loss': 0,
                }
            results[key]['play_game_current_points'] += data['play_game_current_points']
            results[key]['play_game_total_points'] += data['play_game_total_points']
            results[key]['loss'] += data['loss']

    for key in results.keys():
        results[key]['accuracy'] = results[key]['play_game_current_points'] / \
            results[key]['play_game_total_points']

    sorted_dict_desc = dict(
        sorted(results.items(), key=lambda item: item[1]['loss'], reverse=True))

    best_element = next(iter(sorted_dict_desc.items()))
    return best_element


train_file_path = "/Users/raresraf/code/project-martial/samples/rcomplexity/rcomplexity_dataset_results.json"
best_element = run_on_file_path(train_file_path)
print("best train element: ", best_element)

target = 'modules/drivers:rcomplexity_driver'
run_test_bazel_command(target, best_element[0][0],
                       best_element[0][1][0],
                       best_element[0][1][1],
                       best_element[0][1][2],
                       best_element[0][1][3],
                       best_element[0][1][4],
                       best_element[0][1][5],
                       best_element[0][1][6],
                       best_element[0][1][7],
                       best_element[0][1][8],
                       best_element[0][1][9],
                       best_element[0][1][10],
                       best_element[0][1][11],
                       best_element[0][1][12],
                       best_element[0][1][13],
                       best_element[0][1][14],
                       best_element[0][1][15],
                       best_element[0][1][16],
                       best_element[0][1][17],
                       best_element[0][1][18],
                       best_element[0][1][19],
                       best_element[0][1][20],
                       best_element[0][1][21],
                       best_element[0][1][22],
                       best_element[0][1][23],
                       best_element[0][1][24],
                       best_element[0][1][25],
                       best_element[0][1][26],
                       best_element[0][1][27],
                       best_element[0][1][28],
                       best_element[0][1][29],
                       best_element[0][1][30],
                       best_element[0][1][31],
                       best_element[0][1][32],
                       best_element[0][1][33],
                       best_element[0][1][34],
                       best_element[0][1][35],
                       )


test_file_path = "/Users/raresraf/code/project-martial/samples/rcomplexity/test_rcomplexity_dataset_results.json"
best_element = run_on_file_path(test_file_path)
print("best test element: ", best_element)
