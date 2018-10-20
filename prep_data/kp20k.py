import json
import mmap

from typing import List
from textacy import preprocess_text
from utils import apply_parallel, flatten_list
from tqdm import tqdm
import string, re


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def get_text(text):
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
    return processed_text


def process_sentences_constructor():
    """Generate a function that will clean and tokenize text."""
    def process_sentences(lines):
        data = []
        try:
            for line in lines:
                json_line = json.loads(line)
                if 'abstract' in json_line:
                    data.append(get_text(json_line['abstract']))
                if 'title' in json_line:
                    data.append(get_text(json_line['title']))
        except Exception as e:
            print('error '+e)
        return data

    return process_sentences


def parallel_process_text(data: List[str]) -> List[List[str]]:
    """Apply cleaner -> tokenizer."""
    process_text = process_sentences_constructor()
    return flatten_list(apply_parallel(process_text, data))


filename = '/Users/jose.mena/dev/personal/data/kp20k/kp20k/ke20k_training.json'
data = list()
lines = []
with open(filename) as f:
    for line in tqdm(f, total=get_num_lines(filename)):
        lines.append(line)
data = parallel_process_text(lines)
dest_filename = '/Users/jose.mena/dev/personal/data/kp20k/kp20k_multiline.txt'
with open(dest_filename, 'w') as f:
    for text in data:
        f.write(text+'\n')
