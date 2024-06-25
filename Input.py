import os
from PIL import Image

class Input():

    def __init__(self, image, caption1, caption2, article_url):
        self.image = image
        self.caption1 = caption1
        self.caption2 = caption2
        self.article_url = article_url
    
    def is_valid(self, kind):
        return self.image and self.caption1 if kind == 0 or kind == 2 else self.image

    def __str__(self):
        return f"image: {self.image}, caption1: {self.caption1}, caption2: {self.caption2}, article_url: {self.article_url}"
    
    def get_image(self):
        return Image.open(self.image)
    
    def get_image_name(self):
        return self.image.name