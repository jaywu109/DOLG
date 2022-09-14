#!/usr/bin/env python
""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import fire
import os
import lmdb
import pandas as pd

from tqdm import tqdm
from PIL import Image


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(input_root, input_csv, output_path, check_valid=True, dataset='landmark'):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        input_root  : input folder root path where store the images
        input_csv   : input csv storing the ground truth
        output_path : LMDB output path
        check_valid : if true, check the validity of every image
    """
    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(output_path, map_size=1099511627776)
    cache = {}
    cnt = 1

    df = pd.read_csv(input_csv)
    imgdb = {}
    for _, (filename, label) in df.iterrows():
        if dataset == 'landmark':
            image_path = os.path.join(input_root, os.sep.join(list(filename)[:3]), filename + '.jpg')
        elif dataset == 'consumer':
            image_path = filename
        else:
            raise ValueError('Does not support {}'.format(dataset))
        if not os.path.isfile(image_path):
            print('Cannot find {}'.format(image_path))
            continue
        imgdb[image_path] = str(label)

    for image_path, label in tqdm(imgdb.items()):

        with open(image_path, 'rb') as f:
            imageBin = f.read()
        if check_valid:
            try:
                Image.open(image_path)
            except Exception:
                print('Cannot load {}'.format(image_path))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        nameKey = 'name-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        cache[nameKey] = os.path.basename(image_path).encode()

        if cnt % 1000 == 0:
            write_cache(env, cache)
            cache = {}
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    write_cache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    fire.Fire(createDataset)
