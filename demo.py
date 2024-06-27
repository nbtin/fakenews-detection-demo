import os
from tabnanny import check
import streamlit as st
from Input import Input
from Config import Config
from Function import Function
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


IMAGE_FOLDER = "trufor-clone/test_docker/images"
ORIGINAL_PATH = "/home/peter/Documents/repositories/fakenews-detection-demo/"


def information():
    st.set_page_config(
        page_title="Fake News Detection Demo",
        page_icon=":newspaper:",
        layout="wide",
    )
    st.html("<h1 style='text-align: center;'>Fake News Detection Demo</h1>")
    st.html(
        "<center><p style='font-weight: bold;'>Student 1: Nguyen Van Loc (20120131) <br> Student 2: Nguyen Bao Tin (20120596)</p></center>"
    )

    st.info(
        ":bulb: This fake news detection demo refers to the following works:\n\n - Cheapfakes Detection: [A Unified Network for Detecting Out-Of-Context Information Using Generative Synthetic Data](https://dl.acm.org/doi/10.1145/3652583.3657599) (ours)\n\n - Deepfakes Detection: [TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization](https://grip-unina.github.io/TruFor/)"
    )


def get_input(config):
    image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    # if image is not None:
    #     l_col, c_col, r_col = st.columns(3)
    #     with c_col:
    #         st.image(image, width=200, clamp=True)
    if config.get_kind() == 1:
        caption1 = st.text_input(
            "Caption 1 (optional): ", placeholder="Enter caption here"
        )
    else:
        caption1 = st.text_input("Caption 1: ", placeholder="Enter caption here")
    caption2 = st.text_input("Caption 2 (optional): ", placeholder="Enter caption here")
    article_url = st.text_input(
        "Article URL (optional): ", placeholder="Enter the article URL here"
    )

    input = Input(image, caption1, caption2, article_url)

    if not input.is_valid(config.get_kind()) and config.get_kind() != 1:
        st.warning(
            ":warning: Please **upload an image** and **enter caption 1** to continue"
        )
    elif not input.is_valid(config.get_kind()) and config.get_kind() == 1:
        st.warning(":warning: Please **upload an image** to continue")
    return input


def sidebar_config():
    st.sidebar.markdown("# Configurations")
    kind = st.sidebar.radio(
        "**Kind of Manipulation**",
        ("Cheapfakes (Ours)", "Manipulated Images (TruFor)", "Both"),
        index=1,
    )

    # st.sidebar.write("You chose: ", kind)
    st.sidebar.write("#### Optional Features")
    roc_value = st.sidebar.checkbox("Reputation Online Checking", value=False)
    roc_info = st.sidebar.info(
        ":bulb: For more information about Reputation Online Checking, please refer to our [paper](https://dl.acm.org/doi/10.1145/3652583.3657599)"
    )
    roc_service = None
    if roc_value:
        roc_info.empty()
        roc_service = st.sidebar.radio("Choose a service", ("Bing", "Google"), index=0)
        # st.sidebar.write("You chose: ", roc_service)

    return Config(kind, roc_value, roc_service)


def save_trufor_all_in_1(origin_list,  check_list):
    for i, file in enumerate(check_list):
        b = np.load(file)
        origin_image = Image.open(origin_list[i])

        fig = plt.figure(figsize=(15, 5))

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(origin_image)
        ax1.set_title("Original Image")

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(b["map"])
        ax2.set_title("Localization Map")

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(b["conf"], cmap="gray")
        ax3.set_title("Confidence Map")

        fig.suptitle(f"Score: {b['score']}", y=0.05)

        plt.savefig(f"results/{input.get_image_name().split('.')[0]}_trufor_result.png")


def save_trufor_sep(origin_list, check_list):
    for i, file in enumerate(check_list):
        b = np.load(file)
        origin_image = Image.open(origin_list[i])

        loc_map = b["map"]
        # save the localization map to image
        plt.imsave(
            f"results/{input.get_image_name().split('.')[0]}_loc_map.png", loc_map
        )

        conf_map = b["conf"]
        # save the confidence map to image
        plt.imsave(
            f"results/{input.get_image_name().split('.')[0]}_conf_map.png",
            conf_map,
            cmap="gray",
        )

        with open(
            f"results/{input.get_image_name().split('.')[0]}_trufor_result.txt", "w"
        ) as f:
            f.write(f"{b['score']}")


def run(input, config):
    # get current directory
    current_dir = os.getcwd()
    input_image_folder_path = os.path.join(current_dir, IMAGE_FOLDER)
    for f in os.listdir(input_image_folder_path):
        os.remove(os.path.join(input_image_folder_path, f))
    input.get_image().save(
        os.path.join(IMAGE_FOLDER, input.get_image_name())
    )  # save image to images folder in docker folder
    input.get_image().save(
        os.path.join("input_images", input.get_image_name())
    )  # save image to images folder in pipeline folder
    if config.kind == "Manipulated Images (TruFor)":
        os.chdir("./trufor-clone/test_docker")
        os.system("bash docker_run.sh")
        os.chdir(ORIGINAL_PATH)
        origin_list = [
            "./trufor-clone/test_docker/images/" + input.get_image_name(),
        ]

        check_list = [
            "./trufor-clone/test_docker/output/" + input.get_image_name() + ".npz",
        ]

        # check if file exists
        if not os.path.isfile(check_list[0]):
            print("File does not exist")
            return

        save_trufor_all_in_1(origin_list, check_list)
    else:
        pass


def show_results_all_in_1(input, kind):
    if kind == 1:
        st.header("Results")
        st.image(
            f"results/{input.get_image_name().split('.')[0]}_trufor_result.png",
            use_column_width=True,
        )

def result_trufor(input):
    st.header("Results")
    origin, loc_map, conf_map = st.columns(3)

    with origin:
        st.image(
            f"input_images/{input.get_image_name()}",
            caption="Original Image",
            use_column_width=True,
        )

    with loc_map:
        st.image(
            f"results/{input.get_image_name().split('.')[0]}_loc_map.png",
            caption="Localization Map",
            use_column_width=True,
        )

    with conf_map:
        st.image(
            f"results/{input.get_image_name().split('.')[0]}_conf_map.png",
            caption="Confidence Map",
            use_column_width=True,
        )

    with open(
        f"results/{input.get_image_name().split('.')[0]}_trufor_result.txt", "r"
    ) as f:
        score = f.read()

    # Create a placeholder for the progress bar and text
    progress_bar = st.empty()
    progress_text = st.empty()

    # Calculate the percentage for the progress bar
    percentage = float(score)

    # Update the progress bar with the current percentage
    progress_bar.progress(
        percentage, text=f"Integrity Score of Image: {float(score):.4f}"
    )

    # Update the text under the progress bar to show min, current, and max values
    progress_style = f"""
    <div style="
        display: flex;
        justify-content: space-between;
        align-items: center;">
        <span>0 (Pristine)</span>
        <span style="position: absolute; left: {percentage * 100}%; transform: translateX(-50%);"></span>
        <span>1 (Manipulated)</span>
    </div>
    """
    progress_text.markdown(progress_style, unsafe_allow_html=True)
    st.balloons()

def result_cheapfakes(input):
    pass

def result_hybrid(input):
    pass

def result_both(input):
    pass

def show_results_sep(input, kind):
    if kind == 0:
        result_cheapfakes(input)
    elif kind == 1:
        result_trufor(input)
    else:
        result_both(input)


if __name__ == "__main__":
    # create the directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("input_images"):
        os.makedirs("input_images")
    information()
    config = sidebar_config()
    # print(config)
    input = get_input(config)
    # print(input)
    not_implemented_warning = None
    # print(config.get_kind())
    if input.is_valid(config.get_kind()):
        submitted = st.button("Submit", on_click=run, args=(input, config))
        if not Function(config.get_kind()).is_available():
            not_implemented_warning = st.warning(
                "This function is not implemented yet :sob:"
            )

        if submitted and Function(config.get_kind()).is_available():
            if not_implemented_warning is not None:
                not_implemented_warning.empty()
            show_results_sep(input, config.get_kind())
