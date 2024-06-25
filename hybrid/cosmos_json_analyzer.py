import json


with open("false_predictions_cosmos.json", "r") as f:
    false_predictions = json.load(f)

with open("false_predictions_cosmos.json", "w") as f:
    json.dump(false_predictions, f, indent=4)

processed_cosmos_false_predictions = {}
for prediction in false_predictions:
    processed_cosmos_false_predictions[prediction["img_path"]] = [
        prediction["pred_context_cosmos"],
        prediction["actual_context"]
    ]

with open("processed_cosmos_false_predictions.json", "w") as f:
    json.dump(processed_cosmos_false_predictions, f, indent=4)
