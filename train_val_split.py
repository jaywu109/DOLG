#!/usr/bin/env python
import os
import argparse
import pandas as pd
import numpy as np

np.random.seed(0)


def split_train_val(label_path, ratio=0.5, dataset='landmark'):
    label_root = os.path.dirname(label_path)
    label_name = os.path.basename(label_path)
    df = pd.read_csv(label_path, sep=',')

    if dataset == 'landmark':
        val = 'landmark_id'
    elif dataset == 'consumer':
        val = 'id'
    else:
        raise ValueError('Does not support {}'.format(dataset))

    landmark_ids = df[val].unique()

    is_valid = np.random.rand(len(landmark_ids)) < ratio
    train_landmark_ids = set(landmark_ids[~is_valid])
    valid_landmark_ids = set(landmark_ids[is_valid])

    train_df = df[df[val].isin(train_landmark_ids)]
    valid_df = df[df[val].isin(valid_landmark_ids)]

    train_df.to_csv(os.path.join(label_root, 'train_' + label_name), sep=',', index=False)
    valid_df.to_csv(os.path.join(label_root, 'valid_' + label_name), sep=',', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split the label into train/valididation')
    parser.add_argument('label_path', type=str, help='Path of the label')
    parser.add_argument('--ratio', type=float, default=0.5, help='Ratio of the validation set')
    parser.add_argument('--dataset', default='landmark', help='Dataset (landmark, consumer, etc.)')
    args = parser.parse_args()
    split_train_val(args.label_path, args.ratio, args.dataset)
