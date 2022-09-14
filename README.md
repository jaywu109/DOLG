# Pytorch-DOLG

- Pytorch lightning version of the [DOLG](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_DOLG_Single-Stage_Image_Retrieval_With_Deep_Orthogonal_Fusion_of_Local_ICCV_2021_paper.pdf)

## Prepare the dataset

- Google Landmark dataset v2
- Consumer-To-Shop: 
  - `python prepare_consumer_to_shop.py -i /home/data/consumer-to-shop/ -o /home/data/consumer-to-shop/train.csv`
- Aliproduct:
  - `python prepare_aliproduct.py -i /home/data/aliproduct/ -o /home/data/aliproduct/train.csv`

## Steps

- Run `train_val_split` to split the dataset
  - Google Landmark: 
    - `./train_val_split.py /home/data/kaggle/google_landmark_2021/train.csv --ratio 0.05`
  - Consumer-To-Shop: 
    - `./train_val_split.py /home/data/consumer-to-shop/train.csv --ratio 0.05 --dataset consumer`
- Run `create_lmdb_dataset.py` for the train/validation set seperately
  - Google Landmark:
    - `./create_lmdb_dataset.py /home/data/kaggle/google_landmark_2021/train/ /home/data/kaggle/google_landmark_2021/train_train.csv /home/data/kaggle/google_landmark_2021/lmdb/train`
    - `./create_lmdb_dataset.py /home/data/kaggle/google_landmark_2021/train /home/data/kaggle/google_landmark_2021/valid_train.csv /home/data/kaggle/google_landmark_2021/lmdb/val`
  - Consumer-To-Shop:
    - `./create_lmdb_dataset.py /home/data/consumer-to-shop/ /home/data/consumer-to-shop/train_train.csv /home/data/consumer-to-shop/lmdb/train --dataset consumer`
    - `./create_lmdb_dataset.py /home/data/consumer-to-shop/ /home/data/consumer-to-shop/valid_train.csv /home/data/consumer-to-shop/lmdb/val --dataset consumer`
- Run `gen_distribution` generating distribution
  - Google Landmark:
    - `./gen_distribution.py /home/data/kaggle/google_landmark_2021/train.csv --output_dir data/google_landmarks_v2`
  - Consumer-To-Shop:
    - `./gen_distribution.py /home/data/consumer-to-shop/train.csv --output_dir data/consumer_to_shop`
- Run `train.py` start training
  - `./run_train.sh`

## Local feature attention map Visualization

- Please check the [notebook](./visual_local.ipynb)

## Acknowledgement

This repo is based on [official DELG](https://github.com/tensorflow/models/tree/master/research/delf) and [DELG-pytorch](https://github.com/feymanpriv/DELG/). 