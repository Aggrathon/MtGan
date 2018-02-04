"""
    This script downloads and prepares all card data needed for training
"""
import os
import urllib.request
from pathlib import Path
import pandas as pd

DIRECTORY = Path("data")
IMAGES = DIRECTORY / 'images'
DB_FILE = DIRECTORY / 'cards.json'

def download_db(overwrite=False):
    """
        Downloads the card database from http://mtgjson.com
    """
    os.makedirs(DIRECTORY, exist_ok=True)
    path = DIRECTORY / 'AllSets.json'
    if os.path.exists(path) and overwrite:
        os.remove(path)
    if not os.path.exists(path):
        print("Downloading card database")
        urllib.request.urlretrieve("http://mtgjson.com/json/AllSets.json", path)
    if os.path.exists(DB_FILE) and overwrite:
        os.remove(DB_FILE)
    if not os.path.exists(DB_FILE):
        print("Parsing downloaded card database")
        cards = []
        db = pd.read_json(path, encoding='utf-8')
        for i in db:
            cards.extend(db[i]['cards'])
        db = pd.DataFrame(cards)
        db['multiverseid'] = pd.to_numeric(db['multiverseid'], 'raise', 'signed')
        db.dropna(subset=['multiverseid'], inplace=True)
        with open(DB_FILE, 'w', encoding='utf-8') as file:
            db.to_json(file, 'records', force_ascii=False, lines=True)
    print()


def download_images(overwrite=False):
    """
        Downloads card art for all the cards in the database
    """
    db = pd.read_json(DB_FILE, 'records', lines=True, encoding='utf-8')
    os.makedirs(IMAGES, exist_ok=True)
    print("Checking and downloading all card art")
    num = 1
    length = len(db)
    for i in db['multiverseid']:
        id = int(i)
        url = 'http://gatherer.wizards.com/Handlers/Image.ashx?multiverseid=%d&type=card'%id
        path = IMAGES / ('%d.jpg'%id)
        if os.path.exists(path) and overwrite:
            os.remove(path)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)
            print(num, '/', length, end='\r', flush=True)
        num += 1
    print()


if __name__ == "__main__":
    download_db()
    download_images()
