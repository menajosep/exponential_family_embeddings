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


filename = '/Users/jose.mena/dev/personal/data/wiki/enwik9_multiline_raw.txt'
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


with open(filename) as f:
    for line in tqdm(f, total=get_num_lines(filename)):
        data.append(get_text(line))
dest_filename = '/Users/jose.mena/dev/personal/data/wiki/wiki_multiline.txt'
with open(dest_filename, 'w') as f:
    for text in data:
        f.write(text+'\n')
