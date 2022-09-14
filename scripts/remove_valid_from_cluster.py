#!/usr/bin/env python
import os
import pandas as pd


def main():
    valid_df = pd.read_csv('/home/data/kaggle/google_landmark_2021/valid_train.csv')

    cluster_df = pd.read_csv('/home/data/kaggle/google_landmark_2019/label_sub_212843.txt',
                             delimiter=' ',
                             header=None,
                             names=['path', 'landmark_id'])
    cluster_df['id'] = cluster_df['path'].apply(lambda x: os.path.basename(x).split('.')[0])

    valid_id = set(valid_df['id'])

    cluster_df_valid = cluster_df[cluster_df['id'].isin(valid_id)]
    cluster_landmark_valid = set(cluster_df_valid['landmark_id'])
    cluster_df = cluster_df[~cluster_df['landmark_id'].isin(cluster_landmark_valid)]

    cluster_df = cluster_df[['id', 'landmark_id']]

    cluster_df.to_csv('/home/data/kaggle/google_landmark_2019/train_train.csv', index=None)


if __name__ == '__main__':
    main()
