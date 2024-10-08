from argparse import Action
import copy
from distutils.command import upload
from email.mime import image
import json
import os
import re
import time

from click import option
from altair import Key
from bs4 import BeautifulSoup
from matplotlib.dates import WE
from matplotlib.pyplot import show
from regex import B
from requests import head, options
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urljoin
from selenium.webdriver.common.action_chains import ActionChains
import pyperclip
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

from hybrid.file_clipboard import *

# This will set the clipboard to use the file-based method
clipboard = CustomClipboard()

# import data

from tenacity import retry, stop_after_attempt, wait_random_exponential


def get_image_urls_from_bing(
    search_engine_url="https://www.bing.com/images/feed", headless=True
):
    if headless:
        # options = webdriver.ChromeOptions()
        # options.add_argument("headless")
        # options.add_argument("window-size=1920x1080")
        # options.add_argument("disable-gpu")
        # # options.add_argument('no-sandbox')
        # # options.add_argument('disable-dev-shm-usage')
        # options.add_argument(
        #     "user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        # )
        # driver = webdriver.Chrome(
        #     service=Service("/usr/bin/chromedriver"), options=options
        # )

        options = Options()
        options.add_argument('--headless')
        options.add_argument("--window-size=1920,1080")
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    else:
        driver = webdriver.Chrome(service=Service("/usr/bin/chromedriver"))
    driver.get(search_engine_url)

    button = driver.find_element(By.ID, "sb_sbip")

    # Upload image (assuming you have a local image)
    upload_field = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "sb_pastepn"))
    )

    if headless:
        # copy_content = pyperclip.paste()
        copy_content = clipboard.paste()
        print(copy_content)

        ActionChains(driver).move_to_element(button).click(button).perform()

        # driver.save_screenshot("screenshot_clickbtn.png")

        ActionChains(driver).move_to_element(upload_field).click(
            upload_field
        ).send_keys(copy_content).send_keys(".").send_keys(Keys.BACKSPACE).send_keys(
            Keys.ENTER
        ).perform()

        # driver.save_screenshot("screenshot_upload.png")
    else:
        ActionChains(driver).move_to_element(upload_field).click(upload_field).key_down(
            Keys.CONTROL
        ).send_keys("v").perform()

    # implicitly wait for page
    driver.implicitly_wait(10)

    page_with_image_btn = driver.find_elements(By.CLASS_NAME, "t-pim")

    if not page_with_image_btn:
        return []

    ActionChains(driver).move_to_element(page_with_image_btn[0]).click(
        page_with_image_btn[0]
    ).perform()

    result_table = driver.find_element(By.CLASS_NAME, "tab-content")

    ActionChains(driver).move_to_element(result_table).scroll_to_element(
        result_table
    ).perform()

    result_list = driver.find_element(By.CLASS_NAME, "pginlv")
    # print(result_list.get_attribute("innerHTML"))

    # for loop through each li to get the each element from the list
    result_elements = result_list.find_elements(By.CSS_SELECTOR, "div.pginlv > ul > li")
    # print(len(result_elements))

    # Wait for results and extract image URLs (up to 50)
    image_links = []
    for element in result_elements:
        if len(image_links) >= 50:
            break
        soup = BeautifulSoup(element.get_attribute("innerHTML"), "html.parser")

        # Find the <a> tag with class "dfnc"
        a_tag = soup.find("a", class_="dfnc")

        # Extract the href attribute
        if a_tag:
            href = a_tag["href"]
            image_links.append(href)

    # Close the WebDriver
    driver.quit()
    return image_links


# @retry(wait=wait_random_exponential(min=2, max=5), stop=stop_after_attempt(15))
def get_image_urls_from_google(
    search_engine_url="https://www.google.com/", headless=True
):
    # Configure WebDriver (replace with your path if needed)
    # options = webdriver.FirefoxOptions()
    # options.add_argument("--headless")
    # driver = webdriver.Firefox(service=Service("/usr/bin/geckodriver"))
    if headless:
        # options = webdriver.ChromeOptions()
        # options.add_argument("headless")
        # options.add_argument("window-size=1920x1080")
        # options.add_argument("disable-gpu")
        # # options.add_argument('no-sandbox')
        # # options.add_argument('disable-dev-shm-usage')
        # options.add_argument(
        #     "user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        # )
        # driver = webdriver.Chrome(
        #     service=Service("/usr/bin/chromedriver"), options=options
        # )
        options = Options()
        options.add_argument('--headless')
        options.add_argument("--window-size=1920,1080")
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    else:
        driver = webdriver.Chrome(service=Service("/usr/bin/chromedriver"))
    driver.get(search_engine_url)

    # find button by both classname and tag
    button = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "div"))
        and EC.presence_of_element_located((By.CLASS_NAME, "nDcEnd"))
    )

    # print(button.get_attribute("outerHTML"))
    ActionChains(driver).move_to_element(button).click(button).perform()

    # wait for page to load
    time.sleep(0.5)

    # if button:
    # print(button)

    # find field with both class and type
    upload_field = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "input"))
        and EC.presence_of_element_located((By.CLASS_NAME, "cB9M7"))
    )

    if headless:
        # copy_content = pyperclip.paste()
        copy_content = clipboard.paste()
        print(copy_content)

        # Move to the upload_field element and click it
        ActionChains(driver).move_to_element(upload_field).click(
            upload_field
        ).send_keys(copy_content).send_keys(Keys.ENTER).perform()

    else:
        # Move to the upload_field element and click it
        ActionChains(driver).move_to_element(upload_field).click(upload_field).key_down(
            Keys.CONTROL
        ).send_keys("v").key_up(Keys.CONTROL).send_keys(Keys.ENTER).perform()

    # implicitly wait for page
    driver.implicitly_wait(10)
    # driver.save_screenshot("screenshot_gg.png")

    page_with_image_btn = driver.find_elements(By.CLASS_NAME, "ICt2Q")
    # page_with_image_btn = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "ICt2Q")))

    # driver.save_screenshot("screenshot2_gg.png")
    if not page_with_image_btn:
        return []

    ActionChains(driver).move_to_element(page_with_image_btn[0]).click(
        page_with_image_btn[0]
    ).perform()

    # show_more_button = WebDriverWait(driver, 10).until(
    #     EC.presence_of_element_located((By.CLASS_NAME, "z5zkXd"))
    # )
    # print(show_more_button.get_attribute("innerHTML"))

    # counting = 0
    # while counting < 3:
    #     # show_more_button = WebDriverWait(driver, 10).until(
    #     #     EC.presence_of_element_located((By.CLASS_NAME, "z5zkXd"))
    #     # )
    #     ActionChains(driver).move_to_element(show_more_button).perform()
    #     ActionChains(driver).scroll_by_amount(0, 500).move_to_element(show_more_button).click(
    #         show_more_button
    #     ).perform()
    #     counting += 1
    #     driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    #     time.sleep(2)

    try:
        result_list = driver.find_element(By.CLASS_NAME, "dg5SXe")
    except:
        return []

    result_elements = result_list.find_elements(By.CSS_SELECTOR, "ul > li")

    # Wait for results and extract image URLs (up to 50)
    image_links = []
    for element in result_elements:
        if len(image_links) >= 50:
            break
        soup = BeautifulSoup(element.get_attribute("innerHTML"), "html.parser")

        a_tag = soup.find("a")

        # Extract the href attribute
        if a_tag:
            href = a_tag["href"]
            image_links.append(href)

    # Close the WebDriver
    driver.quit()
    return image_links


if __name__ == "__main__":
    # Example usage
    # image_path = "https://images.axios.com/5kpawBORcn8PseOV9KS3nNzL13g=/0x0:4000x2250/1920x1080/2024/04/08/1712575249252.jpg"
    image_path = "https://storage.googleapis.com/fakenews-4048f.appspot.com/input_images/159215939.jpg"

    # pyperclip.copy(image_path)
    clipboard.copy(image_path)
    # results = get_image_urls_from_bing(headless=True)
    results = get_image_urls_from_google(headless=True)

    if results:
        print(f"Found {len(results)} image URLs:")
        for url in results:
            print(url)
    else:
        print("No image URLs found.")

    # write to json file
    with open("image_urls_test.json", "w") as f:
        json.dump(results, f)
