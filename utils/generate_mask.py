import argparse
import random
import math
import os
from PIL import Image, ImageDraw

def random_rectangle(size, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
    area = size * size
    for _ in range(100):
        mask_area = random.uniform(scale[0], scale[1]) * area
        aspect_ratio = random.uniform(ratio[0], ratio[1])

        h = int(round(math.sqrt(mask_area * aspect_ratio)))
        w = int(round(math.sqrt(mask_area / aspect_ratio)))

        if h < size and w < size:
            i = random.randint(0, size - h)
            j = random.randint(0, size - w)

            return i, j, h, w
    return 0, 0, size, size

def get_parser():
    parser = argparse.ArgumentParser(
        prog='Generate Mask Data',
        usage='python3 --image_size 256 -n 10000 generate_mask.py',
        description='This module demonstrates generating mask data.',
        add_help=True)

    parser.add_argument('-s', '--image_size', type=int, default=256)
    parser.add_argument('--save_dir', type=str, default='../dataset/mask')
    parser.add_argument('-n', '--n', type=int, default=10000)
    return parser


def main(parser):

    if not os.path.exists(parser.save_dir):
        os.makedirs(parser.save_dir)
    

    for i in range(parser.n):
        x, y, h, w = random_rectangle(parser.image_size)

        mask = Image.new('L', (parser.image_size, parser.image_size), color=255)
        draw = ImageDraw.Draw(mask)
        draw.rectangle((x, y, h, w), fill=0)
        print("save:", i)
        mask.save('{:s}/{:06d}.jpg'.format(parser.save_dir, i))

if __name__ == "__main__":
    parser = get_parser().parse_args()
    main(parser)
