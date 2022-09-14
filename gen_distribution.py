#!/usr/bin/env python
import os
import json
import fire
import pandas as pd
from collections import Counter


def gen_mapping(train_csv, output_dir='data'):
    """Generate mapping for training data
    Args:
        train_csv (str): path of the training csv
        output_dir (str): path of the output directory. (Default: ./data)
    """
    counter = Counter()
    weighting = dict()

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(train_csv)
    content = []
    for _, (filename, label) in df.iterrows():
        content.append({'classname': str(label), 'imagename': filename + '.jpg'})

    print('Total size: ', len(content))

    for item in content:
        counter[item['classname']] += 1

    print('Total number of classes: ', len(counter))

    mapping = dict(zip(counter.keys(), range(len(counter))))
    mapping_path = os.path.join(output_dir, 'mapping.json')
    with open(mapping_path, 'w', encoding='utf-8-sig') as f:
        json.dump(mapping, f, indent=4, ensure_ascii=False)
    print('Class mapping is stored at {}'.format(mapping_path))

    distribution_path = os.path.join(output_dir, 'distribution.json')
    with open(distribution_path, 'w', encoding='utf-8-sig') as f:
        json.dump(counter, f, indent=4, ensure_ascii=False)
    print('Distribution is stored at {}'.format(distribution_path))

    for item in content:
        weighting[item['imagename']] = 1 / counter[item['classname']]

    sample_weights_path = os.path.join(output_dir, 'weighting.json')
    with open(sample_weights_path, 'w') as f:
        json.dump(weighting, f, indent=4, sort_keys=True)
    print('Sample weights is stored at {}'.format(sample_weights_path))


if __name__ == '__main__':
    fire.Fire(gen_mapping)
