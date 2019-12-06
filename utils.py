import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

CHAR_VECTOR = """ !"#$%&'()*+,-./0123456789:;<=>?@[\]_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ{}"""


def gaussian(img):
    row, col, ch = img.shape
    mean = 0
    var = 2
    sigma = var ** 2
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img + gauss
    return noisy


def gen_word(max_len=25):
    text_len = np.random.randint(1, max_len)
    word = ''.join([CHAR_VECTOR[np.random.randint(0, len(CHAR_VECTOR) - 1)] for i in range(text_len)])
    return word


def gen_num(max_len=25):
    text_len = np.random.randint(1, max_len)
    word = ''.join([str(np.random.randint(0, 10)) for i in range(text_len)])
    return word


def clean_word(word):
    # remove citations
    word = re.sub(r'\[[0-9]+\]', '', word)
    # remove non-ascii chars
    word = re.sub(r'[^\x00-\x7F]', '', word)
    return word.strip()


def flash(fn, args_list, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fn, args): args for args in args_list}
        futures = as_completed(future_to_url)
        return futures
