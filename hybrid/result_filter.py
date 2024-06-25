import json

PREDICTION_DIRECTORY = "prediction_results/"

cosmos_false_predictions = {}
with open(
    PREDICTION_DIRECTORY + "processed_cosmos_false_predictions.json",
    "r",
) as f:
    cosmos_false_predictions = json.load(f)
# print(cosmos_false_predictions.keys())
print(len(cosmos_false_predictions))

end_to_end_result = {}
with open(
    PREDICTION_DIRECTORY + "end2end_result_0.845.json",
    "r",
) as f:
    end_to_end_result = json.load(f)
# print(end_to_end_result.keys())
print(len(end_to_end_result))

hybrid_result = {}
with open(
    PREDICTION_DIRECTORY + "hyprid_rersult_0.849.json",
    "r",
) as f:
    hybrid_result = json.load(f)
# print(hybrid_result.keys())
print(len(hybrid_result))


# remove "test/" in keys of cosmos_false_predictions
new_cosmos_false_predictions = {}
for key in cosmos_false_predictions.keys():
    new_key = key.replace("test/", "")
    new_cosmos_false_predictions[new_key] = cosmos_false_predictions[key]
print(new_cosmos_false_predictions.keys())
print(len(new_cosmos_false_predictions))

filtered_result = {}
for key in new_cosmos_false_predictions.keys():
    end_to_end_prediction = end_to_end_result[key]
    hybrid_prediction = hybrid_result[key]
    if (
        end_to_end_prediction[0] != end_to_end_prediction[1]
        and hybrid_prediction[0] != hybrid_prediction[1]
    ):
        continue
    filtered_result[key] = {
        "cosmos": new_cosmos_false_predictions[key],
        "end_to_end": end_to_end_prediction,
        "hybrid": hybrid_prediction,
    }
print(filtered_result.keys())
print(len(filtered_result))
with open("prediction_results/filtered_result.json", "w") as f:
    json.dump(filtered_result, f, indent=4)


filtered_result_cosmos_end_to_end = {}
for key in new_cosmos_false_predictions.keys():
    end_to_end_prediction = end_to_end_result[key]
    hybrid_prediction = hybrid_result[key]
    if end_to_end_prediction[0] != end_to_end_prediction[1]:
        continue
    filtered_result_cosmos_end_to_end[key] = {
        "cosmos": new_cosmos_false_predictions[key],
        "end_to_end": end_to_end_prediction,
    }
print(filtered_result_cosmos_end_to_end.keys())
print(len(filtered_result_cosmos_end_to_end))
with open(
    "prediction_results/filtered_result_cosmos_end_to_end.json",
    "w",
) as f:
    json.dump(filtered_result_cosmos_end_to_end, f, indent=4)

filtered_result_cosmos_hybrid = {}
for key in new_cosmos_false_predictions.keys():
    end_to_end_prediction = end_to_end_result[key]
    hybrid_prediction = hybrid_result[key]
    if hybrid_prediction[0] != hybrid_prediction[1]:
        continue
    filtered_result_cosmos_hybrid[key] = {
        "cosmos": new_cosmos_false_predictions[key],
        "end_to_end": end_to_end_prediction,
    }
print(filtered_result_cosmos_hybrid.keys())
print(len(filtered_result_cosmos_hybrid))
with open(
    "prediction_results/filtered_result_cosmos_hybrid.json",
    "w",
) as f:
    json.dump(filtered_result_cosmos_hybrid, f, indent=4)

filtered_result_end_to_end_hybrid = {}
for key in new_cosmos_false_predictions.keys():
    end_to_end_prediction = end_to_end_result[key]
    hybrid_prediction = hybrid_result[key]
    if (
        end_to_end_prediction[0] == end_to_end_prediction[1]
        or hybrid_prediction[0] != hybrid_prediction[1]
    ):
        continue
    filtered_result_end_to_end_hybrid[key] = {
        "end_to_end": end_to_end_prediction,
        "hybrid": hybrid_prediction,
    }
print(filtered_result_end_to_end_hybrid.keys())
print(len(filtered_result_end_to_end_hybrid))
with open(
    "prediction_results/filtered_result_end_to_end_hybrid.json",
    "w",
) as f:
    json.dump(filtered_result_end_to_end_hybrid, f, indent=4)
