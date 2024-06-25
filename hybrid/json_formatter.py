import json

file_name = "prediction_results/false_predictions_e2e_hybrid_analyzer.json"

with open(file_name) as file:
    result = json.load(file)

print(len(result))

if isinstance(result, list):
    result = {str(i): d for i, d in enumerate(result)}


new_dict = {}
for d in result.values():
    k = d["img_path"].replace("test/", "")
    new_dict[k] = d

with open(file_name, "w") as file:
    json.dump(new_dict, file, indent=4, sort_keys=True)
