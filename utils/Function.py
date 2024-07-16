from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

import pyperclip
from hybrid.selenium_image_searcher import get_image_urls_from_bing as bing_search
from hybrid.selenium_image_searcher import get_image_urls_from_google as google_search
# from hybrid.new_inference_batch_task1 import search_images, check_famous, FAMOUS_PAGES, CONTEXT

FAMOUS_PAGES = [
    "bbc.com",
    "nytimes.com",
    "arabnews.com",
    "reuters.com",
    "sabcnews.com",
    "pbs.org",
    "nbclosangeles.com",
    "apnews.com",
    "news.sky.com",
    "telegraph.co.uk",
    "time.com",
    "denverpost.com",
    "washingtonpost.com",
    "cbc.ca",
    "theguardian.com",
    "pressherald.com",
    "independent.co.uk",
    "gazette.com",
]

from tqdm import tqdm
import json
import os

# singleton class
class Function:
    _instance = None
    kind = None

    def __new__(cls, kind=None):
        if cls._instance is None:
            cls._instance = super(Function, cls).__new__(cls)
            cls.kind = kind
        return cls._instance

    def is_available(self):
        if self.kind == 0:
            return True
        elif self.kind == 1:
            return True
        else:
            return False


class Context:
    """
    EN: The Context defines the interface of interest to clients.
    """

    def __init__(self, strategy: Strategy) -> None:
        """
        EN: Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._strategy = strategy

    @property
    def strategy(self) -> Strategy:
        """
        EN: The Context maintains a reference to one of the Strategy objects.
        The Context does not know the concrete class of a strategy. It should
        work with all strategies via the Strategy interface.
        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        """
        EN: Usually, the Context allows replacing a Strategy object at runtime.
        """

        self._strategy = strategy

    def reputation_online_checking(self, input, headless=True) -> None:
        """
        EN: The Context delegates some work to the Strategy object instead of
        implementing multiple versions of the algorithm on its own.
        """

        # ...

        print("Context: Performing image search...")
        print("Checking if the images are from famous pages...")
        pyperclip.copy(input.get_image_url())
        results = self._strategy.search(headless=headless)
        print(len(results))
        if results:
            print(f"Found {len(results)} image URLs:")
            for url in results:
                print(url)
        else:
            print("No image URLs found.")
        return results
        # ...
    def check_famous(self, input, results):
        print("Checking if the images are from famous pages...")
        clone_results = []
        for result in results:
            clone_results.append(result)

        if input.is_have_article_url():
            clone_results.append(input.get_article_url())

        famous = []
        for url in tqdm(clone_results):
            for page in FAMOUS_PAGES:
                if page in url:
                    famous.append(url)
                    break
        return famous


class Strategy(ABC):
    """
    EN: The Strategy interface declares operations common to all supported
    versions of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    @abstractmethod
    def search(self, headless=True) -> List:
        pass


"""
EN: Concrete Strategies implement the algorithm while following the base
Strategy interface. The interface makes them interchangeable in the Context.
"""


class Bing(Strategy):
    def search(self, headless=True) -> List:
        return bing_search(headless=headless)


class Google(Strategy):
    def search(self, headless=True) -> List:
        return google_search(headless=headless)

