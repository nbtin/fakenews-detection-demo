import json
import os
import re
from tabnanny import check

from sympy import use
import streamlit as st
from utils.Input import Input
from utils.Config import Config
from utils.Function import Function, Context, Bing, Google
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from firebase_admin import credentials, initialize_app, storage
import pyperclip
import firebase_admin
import time

# Init firebase with your credentials
if not firebase_admin._apps:
    cred = credentials.Certificate("service_account.json")
    initialize_app(cred, {"storageBucket": "fakenews-4048f.appspot.com"})

IMAGE_FOLDER = "/thesis-demo/trufor-clone/test_docker/images"
ORIGINAL_PATH = "/thesis-demo/"
CHEAPFAKES_INPUT_FOLDER = "/thesis-demo/input_images/data"

# check if file with name exists
def download_cheapfakes_checkpoint():
    if not os.path.exists("/thesis-demo/hybrid/checkpoint.best_snli_score_0.9090.pt"):
        os.system("gdown 18IA86NDqR5T8u6zS5RNPSfV7rCmA0pBi")
        os.system(
            "mv /thesis-demo/checkpoint.best_snli_score_0.9090.pt /thesis-demo/hybrid/checkpoint.best_snli_score_0.9090.pt"
        )
    if not os.path.exists("/thesis-demo/trufor-clone/test_docker/weights/"):
        os.system("gdown 1IWvNt6_bvHbYfXpN53XdxLmTPhg7hds0")
        os.system("mv /thesis-demo/TruFor_weights.zip /thesis-demo/trufor-clone/test_docker/TruFor_weights.zip")
        os.system("unzip -q -n /thesis-demo/trufor-clone/test_docker/TruFor_weights.zip -d /thesis-demo/trufor-clone/test_docker/ && rm /thesis-demo/trufor-clone/test_docker/TruFor_weights.zip")



def information():
    st.set_page_config(
        page_title="Thesis Demo on Fake News Analysis & Detection",
        page_icon=":newspaper:",
        layout="wide",
    )
    st.html(
        "<h1 style='text-align: center;'>Thesis Demo on Fake News Analysis & Detection<br>Honors Program 2020</h1>"
    )
    st.html(
        "<center><h5 style='font-weight: bold;'>Student 1: Nguyen Van Loc (20120131) <br> Student 2: Nguyen Bao Tin (20120596)</h5></center>"
    )
    with st.expander("Detail of our works"):
        st.info(
            ":bulb: This fake news detection demo refers to the following works:\n\n - Cheapfakes Detection: \n\n\t - [A Unified Network for Detecting Out-Of-Context Information Using Generative Synthetic Data](https://dl.acm.org/doi/10.1145/3652583.3657599) (ours) - ICMR 2024\n\n\t - [A Hybrid Approach for Cheapfake Detection Using Reputation Checking and End-To-End Network](https://dl.acm.org/doi/10.1145/3660512.3665521) (ours) - SCID 2024\n\n - Deepfakes Detection: [TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization](https://grip-unina.github.io/TruFor/)"
        )

# check uploaded image size (w x h) is satisfied or not
def is_able_to_run(image):
    if image is not None:
        image_check = Image.open(image)
        width, height = image_check.size
        return width <= 1080 and height <= 720
    return False


def get_input(config):
    l_col, c1_col, c2_col, r_col = st.columns([0.8, 0.02, 0.16, 0.02])
    with l_col:
        image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        while image is not None and not is_able_to_run(image):
            st.toast("You've reached the maximum image size.", icon="😢")
            time.sleep(1)
            st.toast("Try again with a size of 1080x720 or smaller!", icon="🤗")
            image = None
    with c2_col:
        placeholder = st.image(
            Image.open("/thesis-demo/assets/placeholder.png"), use_column_width=True
        )
    if image is not None:
        placeholder.empty()
        with c2_col:
            placeholder = st.image(image, use_column_width=True)
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
        index=0,
    )

    # st.sidebar.write("You chose: ", kind)
    st.sidebar.write("#### Optional Features")
    if kind != "Manipulated Images (TruFor)":
        roc_value = st.sidebar.checkbox("Reputation Online Checking (ROC)", value=False)
        roc_info = st.sidebar.info(
            ":bulb: For more information about Reputation Online Checking, please refer to our [paper](https://dl.acm.org/doi/10.1145/3652583.3657599)"
        )
        roc_service = None
        if roc_value:
            roc_info.empty()
            roc_service = st.sidebar.radio(
                "Choose a service", ("Google", "Bing"), index=0
            )
            # st.sidebar.write("You chose: ", roc_service)
    else:
        roc_value = False
        roc_service = None
    return Config(kind, roc_value, roc_service)


def save_trufor_all_in_1(origin_list, check_list):
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

        plt.savefig(
            f"/thesis-demo/results/{input.get_image_name().split('.')[0]}_trufor_result.png"
        )


def save_trufor_sep(origin_list, check_list):
    for i, file in enumerate(check_list):
        b = np.load(file)
        origin_image = Image.open(origin_list[i])

        loc_map = b["map"]
        # save the localization map to image
        plt.imsave(
            f"/thesis-demo/results/{input.get_image_name().split('.')[0]}_loc_map.png",
            loc_map,
        )

        conf_map = b["conf"]
        # save the confidence map to image
        plt.imsave(
            f"/thesis-demo/results/{input.get_image_name().split('.')[0]}_conf_map.png",
            conf_map,
            cmap="gray",
        )

        with open(
            f"/thesis-demo/results/{input.get_image_name().split('.')[0]}_trufor_result.txt",
            "w",
        ) as f:
            f.write(f"{b['score']}")


def run_trufor(input):
    os.chdir(ORIGINAL_PATH + "trufor-clone/test_docker/src")
    # os.system("bash docker_run.sh")
    os.system("python trufor_test.py")
    origin_list = [
        "/thesis-demo/trufor-clone/test_docker/images/" + input.get_image_name(),
    ]

    check_list = [
        "/thesis-demo/trufor-clone/test_docker/output/"
        + input.get_image_name()
        + ".npz",
    ]

    # check if file exists
    if not os.path.isfile(check_list[0]):
        print("File does not exist")
        return

    # save_trufor_all_in_1(origin_list, check_list)
    save_trufor_sep(origin_list, check_list)


def run_cheapfakes(input, config):
    if config.get_roc_value():
        print(f"input: {input}")
        print(f"image url: {input.get_image_url()}")
        # pyperclip.copy(input.get_image_url())
        if config.get_roc_service() == "Bing":
            context = Context(Bing())
        else:
            context = Context(Google())
        results = context.reputation_online_checking(input, headless=True)
        famous = context.check_famous(input, results)
        # write each famous element into a text file
        with open(
            f"/thesis-demo/results/{input.get_image_name().split('.')[0]}_roc.txt",
            "w",
        ) as f:
            f.write("\n".join(famous))
    else:
        pass


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

    # Put your local file path
    fileName = os.path.join("input_images", input.get_image_name())
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

    # Opt : if you want to make public access from the URL
    blob.make_public()

    input.update_image_url(blob.public_url)

    if config.kind == "Manipulated Images (TruFor)":
        # # os.chdir("./trufor-clone/test_docker")
        # os.chdir(ORIGINAL_PATH + "trufor-clone/test_docker/src")
        # # os.system("bash docker_run.sh")
        # os.system("python trufor_test.py")
        # origin_list = [
        #     "/thesis-demo/trufor-clone/test_docker/images/" + input.get_image_name(),
        # ]

        # check_list = [
        #     "/thesis-demo/trufor-clone/test_docker/output/"
        #     + input.get_image_name()
        #     + ".npz",
        # ]

        # # check if file exists
        # if not os.path.isfile(check_list[0]):
        #     print("File does not exist")
        #     return

        # # save_trufor_all_in_1(origin_list, check_list)
        # save_trufor_sep(origin_list, check_list)
        run_trufor(input)
    elif config.kind == "Cheapfakes (Ours)":
        # if config.get_roc_value():
        #     print(f"input: {input}")
        #     print(f"image url: {input.get_image_url()}")
        #     # pyperclip.copy(input.get_image_url())
        #     if config.get_roc_service() == "Bing":
        #         context = Context(Bing())
        #     else:
        #         context = Context(Google())
        #     results = context.reputation_online_checking(input, headless=True)
        #     famous = context.check_famous(input, results)
        #     # write each famous element into a text file
        #     with open(
        #         f"/thesis-demo/results/{input.get_image_name().split('.')[0]}_roc.txt",
        #         "w",
        #     ) as f:
        #         f.write("\n".join(famous))
        # else:
        #     pass
        run_cheapfakes(input, config)
    else:
        run_trufor(input)
        run_cheapfakes(input, config)


def show_results_all_in_1(input, kind):
    if kind == 1:
        st.header("Results")
        st.image(
            f"/thesis-demo/results/{input.get_image_name().split('.')[0]}_trufor_result.png",
            use_column_width=True,
        )


def result_trufor(input):
    st.html("<h2 style='text-align: center;'>Manipulated Image Checking Results</h2>")
    origin, loc_map, conf_map = st.columns(3)

    with origin:
        st.image(
            f"/thesis-demo/input_images/{input.get_image_name()}",
            caption="Original Image",
            use_column_width=True,
        )

    with loc_map:
        st.image(
            f"/thesis-demo/results/{input.get_image_name().split('.')[0]}_loc_map.png",
            caption="Localization Map",
            use_column_width=True,
        )

    with conf_map:
        st.image(
            f"/thesis-demo/results/{input.get_image_name().split('.')[0]}_conf_map.png",
            caption="Confidence Map",
            use_column_width=True,
        )

    with open(
        f"/thesis-demo/results/{input.get_image_name().split('.')[0]}_trufor_result.txt",
        "r",
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
    # st.balloons()

    os.chdir(ORIGINAL_PATH)


def result_roc(input):
    st.html("<h2 style='text-align: center;'>ROC Result</h2>")
    with open(
        f"/thesis-demo/results/{input.get_image_name().split('.')[0]}_roc.txt", "r"
    ) as f:
        famous = f.readlines()
    famous_size = len(famous)
    if famous_size == 0:
        print("Cannot find the image from prestigious sources")
        st.warning("Cannot find the image from prestigious sources")
    else:
        print(f"We found {famous_size} sources with the provided image: {famous}")
        if famous_size == 1:
            st.html(
                f"<h4>Found {famous_size} prestigious source with provided image</h4>"
            )
        else:
            st.html(
                f"<h4>Found {famous_size} prestigious sources with provided image</h4>"
            )
        for i, source in enumerate(famous):
            st.write(f"{i + 1}. {source}")
    os.chdir(ORIGINAL_PATH)


def result_cheapfakes(input, hybrid=False):
    # if hybrid:
    #     result_roc(input)

    # st.warning("Not implemented yet :))")
    # check if cheapfakes input folder exist
    if not os.path.exists(CHEAPFAKES_INPUT_FOLDER):
        os.makedirs(CHEAPFAKES_INPUT_FOLDER)

    if not os.path.exists(CHEAPFAKES_INPUT_FOLDER + "/test/"):
        os.makedirs(CHEAPFAKES_INPUT_FOLDER + "/test/")

    # save input into folder for running cheapfakes detection
    # create a json file to store metadata of input
    with open(f"{CHEAPFAKES_INPUT_FOLDER}/test.json", "w") as f:
        input_data = {
            "img_local_path": f"test/{input.get_image_name()}",
            "caption1": input.get_caption1(),
            "caption2": input.get_caption2(),
            "context_label": "1",
            "article_url": (
                input.get_article_url() if input.is_have_article_url() else ""
            ),
        }
        # write the input_data into json file in json format
        json.dump(input_data, f)

    # save the image to the folder
    input.get_image().save(f"{CHEAPFAKES_INPUT_FOLDER}/test/{input.get_image_name()}")

    # run cheapfakes detection
    # export environment variable
    os.environ["INPUT_FOLDER"] = CHEAPFAKES_INPUT_FOLDER
    os.environ["DATA"] = CHEAPFAKES_INPUT_FOLDER + "/test.json"

    import subprocess

    # Change directory
    print("im here")
    # os.chdir(ORIGINAL_PATH + "cheapfakes_detection_SCID2024/run_scripts/snli_ve")
    os.chdir(ORIGINAL_PATH + "hybrid/run_scripts/snli_ve")

    # Set environment variables
    os.environ["MASTER_PORT"] = "7091"
    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":../../flops/"
    os.environ["user_dir"] = "../../ofa_module"
    os.environ["bpe_dir"] = "../../utils/BPE"
    os.environ["split"] = "test"
    # os.environ["checkpoint_path"] = (
    #     ORIGINAL_PATH
    #     + "cheapfakes_detection_SCID2024/checkpoint.best_snli_score_0.8290.pt"
    # )
    os.environ["checkpoint_path"] = (
        ORIGINAL_PATH + "hybrid/checkpoint.best_snli_score_0.9090.pt"
    )
    os.environ["result_path"] = "../../results/snli_ve"
    os.environ["selected_cols"] = "0,2,3,4,5"

    # Execute the command
    command = """
    CUDA_VISIBLE_DEVICE=0 python ../../inference_batch_task1.py ${DATA} \
        --path=${checkpoint_path} \
        --user-dir=${user_dir} \
        --task=snli_ve \
        --batch-size=2 \
        --log-format=simple --log-interval=10 \
        --seed=7 \
        --gen-subset=${split} \
        --results-path=${result_path} \
        --fp16 \
        --num-workers=0 \
        --model-overrides="{'data':'${data}','bpe_dir':'${bpe_dir}','selected_cols':'${selected_cols}'}"
    """.strip()

    # Replace environment variables in the command with their actual values
    for key, value in os.environ.items():
        command = command.replace("${" + key + "}", value)

    # Execute the command
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()

    # Optionally, print the output and error streams
    print(stdout.decode())
    if stderr:
        print(stderr.decode())

    # get the result in the file
    result_file = f"{ORIGINAL_PATH}/results/cheapfakes/answer.txt"
    with open(result_file, "r") as f:
        result = f.read()

    print(result)

    # display the result
    st.html("<h2 style='text-align: center;'>Cheapfakes Detection Results</h2>")
    if result == "yes":
        st.info(
            "Not found any out-of-context (OOC) information in the image and captions."
        )
    else:
        st.warning("Found out-of-context (OOC) information in the image and captions!")

    os.chdir(ORIGINAL_PATH)


def result_both(input, roc_value):
    if roc_value:
        l_col, r_col = st.columns([0.5, 0.5])
        with l_col:
            result_cheapfakes(input, hybrid=roc_value)
        with r_col:
            result_roc(input)
    else:
        result_cheapfakes(input)

    result_trufor(input)


def show_results_sep(input, kind, roc_value, roc_service):
    success = False
    if kind == 0:
        try:
            if roc_value:
                l_col, r_col = st.columns([0.5, 0.5])
                with l_col:
                    result_cheapfakes(input, hybrid=roc_value)
                with r_col:
                    result_roc(input)
            else:
                result_cheapfakes(input)
            success = True
        except Exception as e:
            print("Cheapfakes: something went wrong :))")
    elif kind == 1:
        try:
            result_trufor(input)
            success = True
        except Exception as e:
            print("TruFor: something went wrong :))")
    else:
        try:
            result_both(input, roc_value)
            success = True
        except Exception as e:
            print("Both: something went wrong :))")

    if success:
        st.balloons()

if __name__ == "__main__":
    download_cheapfakes_checkpoint()
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
            show_results_sep(
                input,
                config.get_kind(),
                config.get_roc_value(),
                config.get_roc_service(),
            )
