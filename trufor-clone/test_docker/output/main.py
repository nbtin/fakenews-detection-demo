import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

check_list = [
    # "poster.jpg.npz",
    # "1.jpg.npz",
    # "2.jpg.npz",
    # "3.jpg.npz",
    # "4.jpg.npz",
    # "5.jpg.npz",
    # "6.jpg.npz",
    # "7.jpg.npz",
    # "8.jpg.npz",
    # "9.jpg.npz",
    "6224a3ece29c40c2198d.jpg.npz",
    "e773db8d9ffd3da364ec.jpg.npz",
    "88ce2807-796f-4f19-b6ae-cc5e645d3555.jpg.npz",
    "download.jpeg.npz",
]

origin_list = [
    # "../images/poster.jpg",
    # "../images/1.jpg",
    # "../images/2.jpg",
    # "../images/3.jpg",
    # "../images/4.jpg",
    # "../images/5.jpg",
    # "../images/6.jpg",
    # "../images/7.jpg",
    # "../images/8.jpg",
    # "../images/9.jpg",
    "../images/6224a3ece29c40c2198d.jpg",
    "../images/e773db8d9ffd3da364ec.jpg",
    "../images/88ce2807-796f-4f19-b6ae-cc5e645d3555.jpg",
    "../images/download.jpeg",
]

for i, file in enumerate(check_list):
    b = np.load(file)
    origin_image = Image.open(origin_list[i])

    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(origin_image)
    ax1.set_title("Original Image: " + file)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(b["map"])
    ax2.set_title("Localization Map")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(b["conf"], cmap="gray")
    ax3.set_title("Confidence Map")

    fig.suptitle(f"Score: {b['score']}", y=0.05)

    plt.savefig(f"{file}.png")
    plt.show()













# origin_image_index = 0

# for file in check_list:
#     b = np.load(file)

#     print(b)
#     lst = b.files
#     # print(lst)
#     for item in lst:
#         print(item)
#         print(b[item])

#     # visualize heatmap of map and conf in the npz file

#     plt.imshow(b["map"])
#     # plt.imshow(b['conf'])
#     plt.show()



#     origin_image_index += 1