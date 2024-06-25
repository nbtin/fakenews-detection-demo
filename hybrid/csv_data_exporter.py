import os
import pandas as pd
import requests
import cv2
import numpy as np
import json

CSV_DIR = "search_results_csv"

# Get the list of file names in test/ directory
image_list = os.listdir("test/")
image_list = sorted(image_list)
print(image_list)

results = {}


for image in image_list:
    print(f"Processing {image}")
    csv_file = os.path.join(CSV_DIR, image.split(".")[0] + ".csv")
    if not os.path.exists(csv_file):
        print(f"Skipping {image}")
        continue
    # Read the CSV
    df = pd.read_csv(csv_file)
    print(df["hostPageUrl"])
    results[image] = [
        url for url in df["hostPageUrl"].values
    ]
    print(results[image])

# Sort the results by key
print(len(results))

with open("csv_extracted_data.json", "w") as f:
    json.dump(results, f, indent=4)

