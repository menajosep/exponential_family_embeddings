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


filename = '/Users/jose.mena/dev/personal/data/kp20k/kp20k/ke20k_training.json'
data = list()


def get_text(field_name):
    text = json_line[field_name]
    text = preprocess_text(text,
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
    return text


with open(filename) as f:
    counter = 0
    for line in tqdm(f, total=get_num_lines(filename)):
        json_line = json.loads(line)
        data.append(get_text('abstract'))
        data.append(get_text('title'))
        counter += 1
        if counter == 10000:
            break
dest_filename = '/Users/jose.mena/dev/personal/data/kp20k/kp20k_1line_10k.txt'
with open(dest_filename, 'w') as f:
    for text in data:
        f.write(text+'\n')
