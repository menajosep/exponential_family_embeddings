import pandas as pd
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


filename = '/Users/jose.mena/dev/personal/data/economics/Full-Economic-News-DFE-839861_unix2.csv'
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
        print("wrong text:" + text)
        processed_text = ""
    return processed_text


news_df = pd.read_csv(filename)
for index, row in tqdm(news_df.iterrows()):
    data.append(get_text(row['text']))
    data.append(get_text(row['headline']))
dest_filename = '/Users/jose.mena/dev/personal/data/economics/economics_multiline.txt'
with open(dest_filename, 'w') as f:
    for text in data:
        f.write(text + '\n')
