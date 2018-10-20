from os import path
from glob import glob
import json
import mmap
import string
import re
from typing import List
from textacy import preprocess_text
from utils import apply_parallel, flatten_list


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


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
                               no_punct=False,
                               no_contractions=False,
                               no_accents=True)


        degrees_pattern = r"\d+deg[f]*"
        processed_text = re.sub(degrees_pattern, "degrees", processed_text)
        remove = string.punctuation
        remove = remove.replace("-", "")
        remove = remove.replace("/", "")
        pattern = r"[{}]".format(remove)
        processed_text = re.sub(pattern, "", processed_text)
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


def process_sentences_constructor():
    """Generate a function that will clean and tokenize text."""
    def process_sentences(recipes):
        data = []
        try:
            for recipe in recipes:
                if 'instructions' in recipe and recipe['instructions'] is not None:
                    data.append(get_text(recipe['instructions']))
                if 'title' in recipe and recipe['title'] is not None:
                    data.append(get_text(recipe['title']))
        except Exception as e:
            print('error '+e)
        return data

    return process_sentences


def parallel_process_text(data: List[str]) -> List[List[str]]:
    """Apply cleaner -> tokenizer."""
    process_text = process_sentences_constructor()
    return flatten_list(apply_parallel(process_text, data))


path_recipe_box_data = '/Users/jose.mena/dev/personal/data/recipes'
data = list()
recipes = load_recipes()
data = parallel_process_text(recipes)

dest_filename = '/Users/jose.mena/dev/personal/data/recipes/recipes_multiline.txt'
with open(dest_filename, 'w') as f:
    for text in data:
        f.write(text+'\n')
