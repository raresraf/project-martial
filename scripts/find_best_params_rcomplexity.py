import json

file_path = "/Users/raresraf/code/project-martial/samples/rcomplexity/rcomplexity_dataset_results.json"


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
                'play_game_current_points' : 0,
                'play_game_total_points' : 0,
            }
        results[key]['play_game_current_points'] += data['play_game_current_points']
        results[key]['play_game_total_points'] += data['play_game_total_points']

for key in results.keys():
    results[key]['accuracy'] = results[key]['play_game_current_points']/results[key]['play_game_total_points']
    
    
sorted_dict_desc = dict(sorted(results.items(), key=lambda item: item[1]['accuracy'], reverse=True))
# print(sorted_dict_desc)

best_element = next(iter(sorted_dict_desc.items()))
print(best_element)
