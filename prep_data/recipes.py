from os import path
from glob import glob
import json
import mmap

from textacy import preprocess_text
from tqdm import tqdm


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


path_recipe_box_data = '/Users/jose.mena/dev/personal/data/recipes'
data = list()


def get_text(text):
    try:
        processed_text = preprocess_text(text,
                               fix_unicode=True,
                               lowercase=True,
                               transliterate=True,
                               no_urls=True,
                               no_emails=True,
                               no_phone_numbers=True,
                               no_numbers=True,
                               no_currency_symbols=True,
                               no_punct=True,
                               no_contractions=False,
                               no_accents=True)
    except:
        print("wrong text:"+text)
        processed_text = ""
    return processed_text


def load_recipe(filename):
    """Load a single recipe collection from disk
    """
    with open(filename, 'r') as f:
        recipes = json.load(f)
    print('Loaded {:,} recipes from {}'.format(len(recipes), filename))
    return recipes


def load_recipes():
    """Load all recipe collections from disk and combine into single dataset
    """
    recipes = {}
    for filename in glob(path.join(path_recipe_box_data, 'recipes_raw*.json')):
        recipes.update(load_recipe(filename))
    print('Loaded {:,} recipes in total'.format(len(recipes)))
    return list(recipes.values())

recipes = load_recipes()

for recipe in tqdm(recipes):
    if 'instructions' in recipe and recipe['instructions'] is not None:
        data.append(get_text(recipe['instructions']))
    if 'title' in recipe and recipe['title'] is not None:
        data.append(get_text(recipe['title']))
dest_filename = '/Users/jose.mena/dev/personal/data/recipes/recipes_multiline.txt'
with open(dest_filename, 'w') as f:
    for text in data:
        f.write(text+'\n')
