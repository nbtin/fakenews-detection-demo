from argparse import Action
from distutils.command import upload
from email.mime import image
import json
import os
import re
import time
from bs4 import BeautifulSoup
from matplotlib.dates import WE
from matplotlib.pyplot import show
from regex import B
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urljoin
from selenium.webdriver.common.action_chains import ActionChains
import pyperclip

import data

from tenacity import retry, stop_after_attempt, wait_random_exponential


def get_image_urls_from_bing(
    search_engine_url="https://www.bing.com/images/feed?form=HDRSC2",
):
    # Configure WebDriver (replace with your path if needed)
    driver = webdriver.Chrome(service=Service("/usr/bin/chromedriver"))
    driver.get(search_engine_url)

    button = driver.find_element(By.ID, "sb_sbip")

    # Upload image (assuming you have a local image)
    upload_field = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "sb_pastepn"))
    )

    ActionChains(driver).move_to_element(button).click(button).perform()
    ActionChains(driver).move_to_element(upload_field).click(upload_field).key_down(
        Keys.CONTROL
    ).send_keys("v").perform()

    # implicitly wait for page
    driver.implicitly_wait(10)

    page_with_image_btn = driver.find_elements(By.CLASS_NAME, "t-pim")

    # print(page_with_image_btn)

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
    print(result_list.get_attribute("innerHTML"))

    # for loop through each li to get the each element from the list
    result_elements = result_list.find_elements(By.CSS_SELECTOR, "div.pginlv > ul > li")
    print(len(result_elements))

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


@retry(wait=wait_random_exponential(min=2, max=5), stop=stop_after_attempt(15))
def get_image_urls_from_google(image_path, search_engine_url="https://www.google.com/"):
    # Configure WebDriver (replace with your path if needed)
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

    # find field with both class and type
    upload_field = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "input"))
        and EC.presence_of_element_located((By.CLASS_NAME, "cB9M7"))
    )

    # Move to the upload_field element and click it
    ActionChains(driver).move_to_element(upload_field).click(upload_field).key_down(
        Keys.CONTROL
    ).send_keys("v").key_up(Keys.CONTROL).send_keys(Keys.ENTER).perform()

    # implicitly wait for page
    driver.implicitly_wait(10)

    page_with_image_btn = driver.find_elements(By.CLASS_NAME, "ICt2Q")

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
    image_path = "https://images.axios.com/5kpawBORcn8PseOV9KS3nNzL13g=/0x0:4000x2250/1920x1080/2024/04/08/1712575249252.jpg"

    pyperclip.copy(image_path)
    # results = get_image_urls_from_bing()
    results = get_image_urls_from_google()

    if results:
        print("Found image URLs:")
        for url in results:
            print(url)
    else:
        print("No image URLs found.")

    # write to json file
    with open("image_urls_test.json", "w") as f:
        json.dump(results, f)
