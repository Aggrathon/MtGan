"""
    This script downloads and prepares all card data needed for training
"""
import os
import urllib.request
from pathlib import Path
from multiprocessing import Pool
from shutil import copyfile
import csv
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

DIRECTORY = Path("data")
IMAGES = DIRECTORY / 'images'
ART = DIRECTORY / 'art'
DB_FILE = DIRECTORY / 'all_cards.json'
SIZE_FILE = DIRECTORY / 'sizes.csv'
OUT_FILE = DIRECTORY / 'cards.csv'
IMAGE_FILE = DIRECTORY / 'images.txt'
ART_FILE = DIRECTORY / 'art.txt'

def download_db(overwrite=False):
    """
        Downloads the card database from http://mtgjson.com
    """
    os.makedirs(DIRECTORY, exist_ok=True)
    path = DIRECTORY / 'AllSets.json'
    if os.path.exists(path) and overwrite:
        os.remove(path)
    if not os.path.isfile(path):
        print("Downloading card database")
        urllib.request.urlretrieve("http://mtgjson.com/json/AllSets.json", path)
    if os.path.exists(DB_FILE) and overwrite:
        os.remove(DB_FILE)
    if not os.path.isfile(DB_FILE):
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
        if os.path.isfile(path) and overwrite:
            os.remove(path)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
            print(num, '/', length, end='\r', flush=True)
        num += 1
    print()

def _cid_read_size(f):
    return (f,)+Image.open(IMAGES/f).size

def check_image_dimensions(overwrite=False):
    """
        Check to see that all images really have the same resolution
    """
    if overwrite and os.path.exists(SIZE_FILE):
        os.remove(SIZE_FILE)
    if not os.path.isfile(SIZE_FILE):
        sizes = []
        print("Reading images and checking sizes")
        pool = Pool()
        df = pd.DataFrame(pool.map(_cid_read_size, os.listdir(IMAGES)), columns=['name', 'width', 'height'])
        with open(SIZE_FILE, 'w', encoding='utf-8') as file:
            df.to_csv(file, encoding='utf-8')
        print()

def plot_image_dimensions(copy_irregular=False):
    """
        Show plots on the distibution of image sizes
    """
    df = pd.read_csv(SIZE_FILE, encoding='utf-8')
    width = df['width'].mode()[0]
    height = df['height'].mode()[0]
    irregular = df[(abs(df.width - width) > 5) | (abs(df.height - height) > 5)]
    print("Irregular:", len(irregular))
    print('Plotting image sizes')
    plt.subplot(2, 2, 1)
    plt.hist(df['width'])
    plt.ylabel("Widths")
    plt.subplot(2, 2, 3)
    plt.hist(df['height'])
    plt.ylabel("Heights")
    plt.subplot(2, 2, 2)
    plt.hist2d(df['width'], df['height'])
    plt.subplot(2, 2, 4)
    plt.plot(df['width'], df['height'], 'ro')
    plt.show()
    if copy_irregular:
        print("Copying irregular images to a new directory")
        directory = DIRECTORY / "IrregularImages"
        os.makedirs(directory, exist_ok=True)
        for f in irregular['name']:
            copyfile(IMAGES / f, directory / f)
    print()

def cull_cards(overwrite=False):
    """
        Remove special cards and unnecessary data
    """
    if overwrite:
        if os.path.exists(OUT_FILE):
            os.remove(OUT_FILE)
        if os.path.exists(IMAGE_FILE):
            os.remove(IMAGE_FILE)
        if os.path.exists(ART_FILE):
            os.remove(ART_FILE)
    if not os.path.isfile(OUT_FILE) or not os.path.isfile(IMAGE_FILE) or not os.path.isfile(ART_FILE):
        print("Removing special cards and stripping data")
        db = pd.read_json(DB_FILE, 'records', lines=True, encoding='utf-8')
        #Keep only normal cards
        db = db[(db.layout == 'normal')]
        #Keep only cards with normal sizes
        df = pd.read_csv(SIZE_FILE, encoding='utf-8')
        width = df['width'].mode()[0]
        height = df['height'].mode()[0]
        irregular = df[(abs(df.width - width) > 5) | (abs(df.height - height) > 5)]
        mask = db['multiverseid'].isin(irregular['name'].transform(lambda s: float(s[:-4])))
        db = db[~mask]
        #Add image path to data
        db = db.assign(image=db['multiverseid'].transform(lambda f: str(IMAGES / ("%d.jpg"%int(f)))))
        db = db.assign(art=db['multiverseid'].transform(lambda f: str(ART / ("%d.jpg"%int(f)))))
        #Remove unnecessary data
        db = db.filter(items=['name', 'manaCost', 'cmc', 'colors', 'types', 'subtypes', 'rarity',\
            'text', 'flavor', 'power', 'toughness', 'loyalty', 'image', 'art'])
        #Save result
        with open(OUT_FILE, 'w', encoding='utf-8') as file:
            db.to_csv(file, encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
        with open(IMAGE_FILE, 'w', encoding='utf-8') as file:
            db['image'].to_csv(file, index=False, encoding='utf-8')
        with open(ART_FILE, 'w', encoding='utf-8') as file:
            db['art'].to_csv(file, index=False, encoding='utf-8')
        print()

def crop_art(overwrite=False):
    """
    Crop the card images to only show the art
    tl =    (26, 37)       (19, 37)
    tr =    (197, 37)      (203, 37)
    bl =    (26, 168)      (19, 171)
    br =    (197, 168)     (203, 171)
    w =     171            184
    h =     131            134
    """
    os.makedirs(ART, exist_ok=True)
    x1 = 23
    x2 = 199
    y1 = 40
    y2 = 168
    db = pd.read_csv(str(OUT_FILE), encoding='utf-8')
    print("Cropping the art from the cards")
    for i, line in db.iterrows():
        img_path = line['image']
        art_path = line['art']
        if os.path.isfile(img_path):
            if overwrite and os.path.exists(art_path):
                os.remove(art_path)
            if not os.path.exists(art_path):
                img = Image.open(img_path)
                img.crop((23, 40, 199, 168)).convert("RGB").save(art_path, quality=95)
    print()

#TODO: Transform into TfRecords?

 
if __name__ == "__main__":
    download_db()
    download_images()
    check_image_dimensions()
    #plot_image_dimensions()
    cull_cards()
    crop_art()
