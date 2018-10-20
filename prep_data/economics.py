import mmap
import re
import string
from typing import List

import pandas as pd
from textacy import preprocess_text

from utils import apply_parallel, flatten_list
from tqdm import tqdm


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
        print("wrong text:" + text)
        processed_text = ""
    return processed_text


def process_sentences_constructor():
    """Generate a function that will clean and tokenize text."""
    def process_sentences(rows):
        data = []
        try:
            for row in rows:
                if 'text' in row:
                    data.append(get_text(row['text']))
                if 'headline' in row:
                    data.append(get_text(row['headline']))
        except Exception as e:
            print('error '+e)
        return data

    return process_sentences


def parallel_process_text(data: List[str]) -> List[List[str]]:
    """Apply cleaner -> tokenizer."""
    process_text = process_sentences_constructor()
    return flatten_list(apply_parallel(process_text, data))


filename = '/Users/jose.mena/dev/personal/data/economics/Full-Economic-News-DFE-839861_unix2.csv'
data = list()
news_df = pd.read_csv(filename)
rows = []
for index, row in tqdm(news_df.iterrows()):
    rows.append(row)
data = parallel_process_text(rows)
dest_filename = '/Users/jose.mena/dev/personal/data/economics/economics_multiline.txt'
with open(dest_filename, 'w') as f:
    for text in data:
        f.write(text + '\n')
