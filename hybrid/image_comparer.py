import os
import pandas as pd
import requests
import cv2
import numpy as np

CSV_DIR = "search_results_csv"

# Get the list of file names in test/ directory
image_list = os.listdir("test/")
image_list = sorted(image_list)
print(image_list)


for image in image_list:
    print(f"Processing {image}")
    csv_file = os.path.join(CSV_DIR, image.split(".")[0] + ".csv")
    if not os.path.exists(csv_file) or os.path.exists(f"thumbnails/{image}"):
        print(f"Skipping {image}")
        continue
    # Read the CSV
    df = pd.read_csv(csv_file)
    # Get the value of the thumbnailUrl
    thumbnailUrl = df["contentUrl"].values[0]
    print(f"Downloading {thumbnailUrl}")
    # Download the image
    try:
        response = requests.get(thumbnailUrl, timeout=10)
    except Exception as e:
        print(f"Error downloading {thumbnailUrl}")
        continue

    # Save the downloaded_image
    with open(f"thumbnails/{image}", "wb") as f:
        f.write(response.content)
    print(f"Saved {image}")

    # Open saved image, and image, use cv2 to combine them, and save the combined image
    # Open the saved image
    saved_image = cv2.imread(f"thumbnails/{image}")
    # Open the original image
    original_image = cv2.imread(f"test/{image}")
    # Combine the images
    # combined_image = np.hstack([original_image, saved_image])
    # # Save the combined image
    # cv2.imwrite(f"combined_images/{image}", combined_image)
    # print(f"Saved combined image for {image}")

