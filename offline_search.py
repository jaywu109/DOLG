"""Offline search.
It is hard coded for LV project temporarily
"""
import os
from typing import Counter
from PIL import Image
import torch
import numpy as np
import faiss
import shutil
import pandas as pd
import h5py
from metric.utils import find_image
from tqdm import tqdm
from torch.utils.data import DataLoader
from metric.model import BaselineModelInfer
from metric.dataset import ListDataset, ResizeCollateDemo, RawDataset
from detector import YoloV4ScaledDetector
from yolov4_scaled.utils.datasets import LoadImages

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# generate index embedding using API/prebuild_index
INDEX_PATH = '/home/data/lv/raw_images'
INDEX_H5_PATH = 'index_embedding.h5'
TEST_H5_PATH = 'test_embedding.h5'
TEST_PATH = '/home/data/lv/scene'
MODEL_PATH = '/home/gitlocal/ICCV-2021-Google-Landmark-Retrieval/saved_models/dolg_256_mogu_bag_v4_cropped/highest_recall.ckpt'
YOLO_IMG_SIZE = 896
YOLO_MODEL_PATH = '/home/data/models/binary_handbag_detector_lv.pt'
YOLO_CONFIDENCE = 0.4
TWO_STAGE = True
OVERWRITE = True
CROP_SIZE = 256


def _process_index_xlsx(path):
    df = pd.read_excel(path, sheet_name='raw')
    df = df[df['category_id'] >= 0]
    df = df[['filename', 'series']]

    df.set_index('filename', inplace=True)
    df = df[df['series'] != 'mapping_not_found']
    df.dropna(inplace=True)

    df.reset_index(inplace=True)
    # remove pics with >= 2 bags in image
    counter = Counter()
    for _, row in df.iterrows():
        counter[row['filename']] += 1
    valid_filenames = set([k for k, v in counter.items() if v == 1])
    df = df[df['filename'].isin(valid_filenames)]
    df.set_index('filename', inplace=True)
    return df


def load_or_set_index_database(path, h5_path, model, detector, overwriten=False, two_stage=False):
    if os.path.isfile(h5_path) and not overwriten:
        print('Loading the exiting index embedding')
        hf = h5py.File(h5_path, 'r')
        embeddings = hf['embedding'][:]
        img_paths = hf['name'][:]

    elif two_stage:
        df = _process_index_xlsx(os.path.join(INDEX_PATH, 'Annotation_batch1-4+judy5-6.xlsx'))
        filenames = set(df.index)
        dataset = LoadImages(path, filenames)
        image_paths, images = [], []
        for p, img, im0, _ in tqdm(dataset):
            pred = detector.predict(img, im0)
            if len(pred) != 1:
                continue
            for xyxy, _, _ in pred:
                x0, y0, x1, y1 = xyxy.cpu().numpy().astype(int)
                subimg = Image.fromarray(im0[y0:y1, x0:x1])
                images.append(subimg)
                image_paths.append(p)
        dataset = ListDataset(images)

        data_loader = DataLoader(dataset,
                                 num_workers=8,
                                 batch_size=128,
                                 shuffle=False,
                                 collate_fn=ResizeCollateDemo(crop_size=CROP_SIZE, short_size=CROP_SIZE),
                                 pin_memory=True)

        embeddings = []
        with torch.no_grad():
            for img_tensor, _ in tqdm(data_loader):
                embedding, _ = model(img_tensor.to(DEVICE))
                embeddings.append(embedding.cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        img_paths = np.array(image_paths)
        img_paths = np.char.encode(img_paths, encoding='utf-8')
        hf = h5py.File(h5_path, mode='w')
        hf.create_dataset('embedding', data=embeddings, maxshape=(None, 512), dtype=np.float32, compression='gzip', chunks=(1, 512))
        hf.create_dataset('name', data=img_paths, maxshape=(None,), dtype='S1000')

    else:
        df = _process_index_xlsx(os.path.join(INDEX_PATH, 'Annotation_batch1-4+judy5-6.xlsx'))
        filenames = set(df.index)
        image_paths = []
        for _, p in find_image(path):
            if os.path.basename(p) in filenames:
                image_paths.append(p)
        image_paths = sorted(image_paths)
        dataset = ListDataset(image_paths)

        data_loader = DataLoader(dataset,
                                 num_workers=8,
                                 batch_size=128,
                                 shuffle=False,
                                 collate_fn=ResizeCollateDemo(crop_size=CROP_SIZE, short_size=CROP_SIZE),
                                 pin_memory=True)

        embeddings, img_paths = [], []
        with torch.no_grad():
            for img_tensor, img_path in tqdm(data_loader):
                embedding, _ = model(img_tensor.to(DEVICE))
                embeddings.append(embedding.cpu().numpy())
                img_paths.extend(img_path)

        embeddings = np.concatenate(embeddings, axis=0)
        img_paths = np.array(img_paths)
        img_paths = np.char.encode(img_paths, encoding='utf-8')
        hf = h5py.File(h5_path, mode='w')
        hf.create_dataset('embedding', data=embeddings, maxshape=(None, 512), dtype=np.float32, compression='gzip', chunks=(1, 512))
        hf.create_dataset('name', data=img_paths, maxshape=(None,), dtype='S1000')

    img_paths = np.char.decode(img_paths, encoding='utf-8')
    return embeddings, img_paths


def load_or_set_test_database(path, h5_path, model, detector, overwriten=False, two_stage=False):
    if os.path.isfile(h5_path) and not overwriten:
        print('Loading the exiting text embedding')
        hf = h5py.File(h5_path, 'r')
        embeddings = hf['embedding'][:]
        img_paths = hf['name'][:]

    elif two_stage:
        dataset = LoadImages(path)
        image_paths, images = [], []
        for p, img, im0, _ in tqdm(dataset):
            pred = detector.predict(img, im0)
            if len(pred) != 1:
                continue
            for xyxy, _, _ in pred:
                x0, y0, x1, y1 = xyxy.cpu().numpy().astype(int)
                subimg = Image.fromarray(im0[y0:y1, x0:x1])
                images.append(subimg)
                image_paths.append(p)
        dataset = ListDataset(images)

        data_loader = DataLoader(dataset,
                                 num_workers=8,
                                 batch_size=128,
                                 shuffle=False,
                                 collate_fn=ResizeCollateDemo(crop_size=CROP_SIZE, short_size=CROP_SIZE),
                                 pin_memory=True)

        embeddings = []
        with torch.no_grad():
            for img_tensor, _ in tqdm(data_loader):
                embedding, _ = model(img_tensor.to(DEVICE))
                embeddings.append(embedding.cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        img_paths = np.array(image_paths)
        img_paths = np.char.encode(img_paths, encoding='utf-8')
        hf = h5py.File(h5_path, mode='w')
        hf.create_dataset('embedding', data=embeddings, maxshape=(None, 512), dtype=np.float32, compression='gzip', chunks=(1, 512))
        hf.create_dataset('name', data=img_paths, maxshape=(None,), dtype='S1000')

    else:
        dataset = RawDataset(path)

        data_loader = DataLoader(dataset,
                                 num_workers=8,
                                 batch_size=128,
                                 shuffle=False,
                                 collate_fn=ResizeCollateDemo(crop_size=CROP_SIZE, short_size=CROP_SIZE),
                                 pin_memory=True)

        embeddings, img_paths = [], []
        with torch.no_grad():
            for img_tensor, img_path in tqdm(data_loader):
                embedding, _ = model(img_tensor.to(DEVICE))
                embeddings.append(embedding.cpu().numpy())
                img_paths.extend(img_path)

        embeddings = np.concatenate(embeddings, axis=0)
        img_paths = np.array(img_paths)
        img_paths = np.char.encode(img_paths, encoding='utf-8')
        hf = h5py.File(h5_path, mode='w')
        hf.create_dataset('embedding', data=embeddings, maxshape=(None, 512), dtype=np.float32, compression='gzip', chunks=(1, 512))
        hf.create_dataset('name', data=img_paths, maxshape=(None,), dtype='S1000')

    img_paths = np.char.decode(img_paths, encoding='utf-8')
    return embeddings, img_paths


def search_with_faiss_cpu(feat_test, feat_index, topk=5):
    n_dim = feat_index.shape[1]

    cpu_index = faiss.IndexFlatIP(n_dim)
    cpu_index.add(feat_index)

    dists, topk_idx = cpu_index.search(x=feat_test, k=topk)
    cpu_index.reset()
    return dists, topk_idx


def plot_figure(test_image, chosen_images, visual_dir='visual'):
    outdir = os.path.join(visual_dir, os.path.basename(test_image).split('.')[0])
    indexdir = os.path.join(outdir, 'index')
    os.makedirs(indexdir, exist_ok=True)

    shutil.copy(test_image, os.path.join(outdir, os.path.basename(test_image)))
    for i, image in enumerate(chosen_images):
        imname = '{:03d}_{}'.format(i, os.path.basename(image))
        shutil.copy(image, os.path.join(indexdir, imname))


def offline_search(model_path, yolo_model_path, overwriten=False, two_stage=False):
    model = BaselineModelInfer.load_from_checkpoint(model_path, strict=False, embedding_dim=512, cs=1024, global_only=False)
    model.eval().to(DEVICE)

    detector = YoloV4ScaledDetector(yolo_model_path, img_size=YOLO_IMG_SIZE, conf_thres=YOLO_CONFIDENCE)

    # index database
    index_embedding, index_paths = load_or_set_index_database(INDEX_PATH,
                                                              INDEX_H5_PATH,
                                                              model,
                                                              detector,
                                                              overwriten=overwriten,
                                                              two_stage=two_stage)
    print('Index image feature shape: {}'.format(index_embedding.shape))

    # test_database
    test_embedding, test_paths = load_or_set_test_database(TEST_PATH,
                                                           TEST_H5_PATH,
                                                           model,
                                                           detector,
                                                           overwriten=overwriten,
                                                           two_stage=two_stage)
    print('Test image feature shape: {}'.format(test_embedding.shape))

    topk_score, topk_idx = search_with_faiss_cpu(test_embedding, index_embedding, topk=3)

    for row, scores, test_path in zip(topk_idx, topk_score, test_paths):
        nearest_paths = index_paths[row]
        yield test_path, nearest_paths, scores


def load_ground_truth():
    df = pd.read_excel(os.path.join(TEST_PATH, 'detection_with_scene_info_final_1.xlsx'), sheet_name='Sheet1')
    df = df[['pic_name', 'collection']]
    df.rename(columns={'collection': 'series', 'pic_name': 'filename'}, inplace=True)

    # remove pics with >= 2 bags in image
    counter = Counter()
    for _, row in df.iterrows():
        counter[row['filename']] += 1
    valid_filenames = set([k for k, v in counter.items() if v == 1])
    df = df[df['filename'].isin(valid_filenames)]

    return df


def make_prediction(model_path, yolo_model_path, visualize=False, threshold=0.8):
    # read index label
    index_label_df = _process_index_xlsx(os.path.join(INDEX_PATH, 'Annotation_batch1-4+judy5-6.xlsx'))

    result_names, result_series = [], []
    for test_path, nearest_paths, scores in offline_search(model_path, yolo_model_path, overwriten=OVERWRITE, two_stage=TWO_STAGE):
        filtered_paths = nearest_paths[:sum(scores > threshold)]
        if len(filtered_paths) > 0:
            if visualize:
                plot_figure(test_path, filtered_paths)
            # select the first one as prediction
            predict_name = os.path.basename(filtered_paths[0])
            item = index_label_df.loc[predict_name]
            result_series.append(item['series'])
            result_names.append(os.path.basename(test_path))
    df_pred = pd.DataFrame.from_dict({'filename': result_names, 'series': result_series})
    return df_pred


if __name__ == '__main__':
    df_gt = load_ground_truth()
    df_pred = make_prediction(MODEL_PATH, YOLO_MODEL_PATH, visualize=False, threshold=0.3)

    df_gt.to_csv('gt.csv')
    df_pred.to_csv('pred.csv')

    df = df_gt.merge(df_pred, on='filename', how='inner', suffixes=('_gt', '_pred'))
    df['correct'] = df['series_gt'] == df['series_pred']
    precision = np.sum(df['correct']) / len(df)
    print("Precision: {}/{} ({:.2f})".format(np.sum(df['correct']), len(df), precision))
    df = df_gt.merge(df_pred, on='filename', how='left', suffixes=('_gt', '_pred'))
    df['correct'] = df['series_gt'] == df['series_pred']
    recall = np.sum(df['correct']) / len(df)
    print("Recall: {}/{} ({:.2f})".format(np.sum(df['correct']), len(df), recall))
    print('F1: {:.2f}'.format(2 * (precision * recall) / (precision + recall)))
