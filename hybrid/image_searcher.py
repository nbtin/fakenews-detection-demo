import json
import io
from search_engine import SearchByImageService
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import os
from google.oauth2 import service_account
import time

SERVICE_ACCOUNT_FILE = "sunlit-fuze-419806-50ba9cdd1ddd.json"


def search_images(img_path):
    """Search for images using the SearchByImageService."""
    # Load the service account credentials from the JSON key file
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)

    # Initialize the Google Drive service using the credentials
    drive_service = build("drive", "v3", credentials=credentials)

    # Create a file metadata dictionary
    file_metadata = {
        "name": os.path.basename(img_path)
    }
    mimetype = (
        "image/jpeg"
        if img_path.endswith(".jpg") or img_path.endswith(".jpeg")
        else "image/png"
    )

    # Create a media object for the image file
    media = MediaIoBaseUpload(io.BytesIO(open(img_path, "rb").read()), mimetype=mimetype)

    # Upload the image file to Google Drive
    file = drive_service.files().create(body=file_metadata, media_body=media, fields="id").execute()

    # Extract the file ID of the uploaded image
    file_id = file.get("id")

    # Set permissions to allow anyone with the link to view the file
    permission = {"type": "anyone", "role": "reader"}
    drive_service.permissions().create(fileId=file_id, body=permission).execute()

    # Get the URL of the uploaded image
    image_url = "https://drive.google.com/uc?id=" + file_id

    # Initialize the SearchByImageService instance
    search_service = SearchByImageService.get_instance()

    # Search for the image using the SearchByImageService
    data = search_service.search(image_url, limit_page=5)

    # Delete the uploaded image file from Google Drive
    drive_service.files().delete(fileId=file_id).execute()
    return data


# Get the list of file names in test/ directory
image_list = os.listdir("test/")
print(image_list)

with open("search_results.json", "r") as f:
    search_results = json.load(f)

for image in image_list:
    if image in search_results:
        print(f"Skipping {image}")
        continue
    # Sleep for 10 seconds
    time.sleep(10)
    print(f"Searching for {image}")
    image_path = os.path.join("test", image)
    result = search_images(image_path)
    print(result)
    host_page_urls = [data["link"] for data in result]
    search_results[image] = host_page_urls
    # Write to JSON file
    with open("search_results.json", "w") as f:
        json.dump(search_results, f, indent=4)
    # Sleep for 10 seconds
    time.sleep(10)

# with open("search_results.json", "r") as f:
#     search_results = json.load(f)

# print(len(search_results))

