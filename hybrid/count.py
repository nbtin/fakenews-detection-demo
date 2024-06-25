import json

with open('image_urls.json', 'r') as f:
    data = json.load(f)
    count = 0
    for item in data:
        count += 1
    print(count)