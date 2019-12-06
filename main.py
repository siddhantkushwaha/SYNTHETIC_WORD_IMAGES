import os
import re
import math

import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image, ImageOps

from utils import gaussian, gen_num, clean_word, flash

fonts_root = 'fonts'
font_paths = [os.path.join(fonts_root, i)
              for i in os.listdir(fonts_root) if i.endswith('.ttf')]

# image params
w, h = 512, 64
b, g, r, a = 0, 0, 0, 0
color = (b, g, r, a)
x, y = 16, 16


def write_word(i, word):
    # randomized
    font_path = font_paths[i % 2]
    underlined = True if i % 3 == 1 else False
    inverted = True if i % 3 == 1 else False

    font = ImageFont.truetype(font_path, np.random.randint(16, 20))
    img_pil = Image.fromarray(np.ones((h, w, 3), np.uint8) * 255)
    draw = ImageDraw.Draw(img_pil)

    # add text
    draw.text((x, y), word, font=font, fill=color)

    # add underline
    tw, th = draw.textsize(word, font=font)
    if underlined:
        draw.line((x, y + th + 1, x + tw, y + th + 1), fill=color)

    if inverted:
        img_pil = ImageOps.invert(img_pil)

    # rotate slightly
    angle = np.random.uniform(-0.2, 0.2)
    img_pil = img_pil.rotate(angle)

    # crop
    img = np.array(img_pil)[y:y + th + 4, x:x + tw]

    # add gaussian noise to image
    img = gaussian(img)

    print(i, word)

    with open(f"out/img_{i}.txt", 'w') as f:
        f.write(word)
    cv2.imwrite(f"out/img_{i}.png", img)


def text_synth():
    dist = {}
    for root, _, files in os.walk('data'):
        for file in files:
            fn = os.path.join(root, file)
            with open(fn, 'r') as f:
                text = f.read()
            text = re.split(r'\s', text)

            print(fn, len(text))

            for word in text:
                word = clean_word(word)
                if len(word) > 25 or len(word) < 1:
                    continue
                dist[word] = dist.get(word, 0) + 1

    print('Unique words - ', len(dist))
    # sort based on frequency
    sorted_dist = sorted(dist.items(), key=lambda i: i[1], reverse=True)

    new_dist = {}
    rank = len(sorted_dist)
    for k, v in sorted_dist:
        mod_freq = math.ceil(math.log(v, 1.9) * math.log(rank, 1.8))
        if mod_freq > 0:
            new_dist[k] = mod_freq
        rank -= 1

    #  add numbers
    for i in range(50000):
        new_dist[gen_num()] = 1

    print(f'A smoothing distribution, word count updated from {sum(dist.values())} to {sum(new_dist.values())}.')

    all_words = []
    for word in new_dist:
        for i in range(new_dist[word]):
            all_words.append(word)

    np.random.shuffle(all_words)

    args = list(enumerate(all_words, 1))

    # write images
    flash(lambda p: write_word(*p), args, max_workers=6144)


if __name__ == '__main__':
    text_synth()
