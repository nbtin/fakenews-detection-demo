import json

PREDICTION_DIRECTORY = "prediction_results/"

with open(
    PREDICTION_DIRECTORY + "standardized_public_test_acm.json",
    "r",
) as f:
    public_test = json.load(f)

new_public_test = {}
for instance in public_test:
    new_public_test[instance["img_local_path"].replace("test/", "")] = {
        "caption1": instance["caption1"],
        "caption2": instance["caption2"],
        "context_label": instance["context_label"],
        "article_url": instance["article_url"],
    }

with open(
    PREDICTION_DIRECTORY + "new_public_test.json",
    "w",
) as f:
    json.dump(new_public_test, f, indent=4)

LIST_FILES = [
    "filtered_result_cosmos_end_to_end.json",
    "filtered_result_cosmos_hybrid.json",
    "filtered_result_end_to_end_hybrid.json",
]

new_dict = {}
new_data_dict = {}
for file in LIST_FILES:
    new_dict = json.load(open(PREDICTION_DIRECTORY + file, "r"))
    for key in new_dict.keys():
        new_data_dict[key] = new_public_test[key]
    with open(
        PREDICTION_DIRECTORY + file.replace("filtered_result", "filtered_data"),
        "w",
    ) as f:
        json.dump(new_data_dict, f, indent=4)
