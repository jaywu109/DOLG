import os
import sys
import torch
import lmdb
import six
import json
import glob
import pandas as pd
import numpy as np
from PIL import Image
from natsort import natsorted
from torch.utils.data import Dataset
from torchvision import transforms as T
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    resizeCollate_v2_OK = True
except ModuleNotFoundError:
    resizeCollate_v2_OK = False
    print('ResizeCollateV2 is disabled')


class ListDataset(Dataset):

    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.nSamples = len(image_list)
        self.transform = transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img = self.image_list[index]
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.image_list[index]


class RawDataset(Dataset):

    def __init__(self, root, rgb=True, maxnum=None):
        self.root = root
        self.rgb = rgb
        self.maxnum = maxnum
        self.image_path_list = []
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = sorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

        if self.maxnum is not None:
            self.nSamples = min(self.maxnum, self.nSamples)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        try:
            if self.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            else:
                img = Image.open(self.image_path_list[index]).convert('L')
        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.rgb:
                img = Image.new('RGB', (256, 256))
            else:
                img = Image.new('L', (256, 256))

        return (img, self.image_path_list[index])


class RawDatasetTraining(Dataset):

    def __init__(self, root, label_csv_path, read_img=True, label_mapping='data/mapping.json', rgb=True, maxnum=None):
        self.root = root
        self.rgb = rgb
        self.read_img = read_img
        self.maxnum = maxnum

        # create self.image_path_list
        label_df = pd.read_csv(label_csv_path)
        img_id_required = set(label_df['id'])
        self.image_path_list = [
            file_path for file_path in glob.glob(os.path.join(root, '**/*.*'), recursive=True)
            if os.path.basename(file_path).split('.')[0] in img_id_required
        ]

        # create self.label_list
        img_id_to_label = {ids: str(labels) for ids, labels in zip(label_df['id'], label_df['landmark_id'])}
        label_reorder_mapping = json.load(open(label_mapping, encoding='utf-8-sig'))
        self.label_list = [
            label_reorder_mapping[img_id_to_label[os.path.basename(file_path).split('.')[0]]] for file_path in self.image_path_list
        ]

        self.nSamples = len(self.image_path_list)

        if self.maxnum is not None:
            self.nSamples = self.maxnum

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        try:
            if self.read_img:
                if self.rgb:
                    img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
                else:
                    img = Image.open(self.image_path_list[index]).convert('L')
            else:
                img = self.image_path_list[index]
        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.rgb:
                img = Image.new('RGB', (256, 256))
            else:
                img = Image.new('L', (256, 256))

        return (img, self.label_list[index])


class RawDatasetFromList(Dataset):

    def __init__(self, root, csv_path, rgb=True):
        self.root = root
        self.csv_path = csv_path
        self.rgb = rgb

        self.image_path_list = []
        df = pd.read_csv(self.csv_path)
        df = df[['fname', 'x0', 'x1', 'y0', 'y1']]
        for _, row in df.iterrows():
            fpath = os.path.join(self.root, row['fname'])
            self.image_path_list.append([fpath, row['x0'], row['y0'], row['x1'], row['y1']])

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        try:
            path, x0, y0, x1, y1 = self.image_path_list[index]
            if self.rgb:
                img = Image.open(path).crop((x0, y0, x1, y1)).convert('RGB')  # for color image
            else:
                img = Image.open(path).crop((x0, y0, x1, y1)).convert('L')
        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.rgb:
                img = Image.new('RGB', (256, 256))
            else:
                img = Image.new('L', (256, 256))

        return (img, self.image_path_list[index])


class LmdbDataset(Dataset):

    def __init__(self, root, label_mapping='data/mapping.json', read_img=True, rgb=True, maxnum=None):

        self.root = root
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        self.rgb = rgb
        self.read_img = read_img
        self.maxnum = maxnum
        with open(label_mapping, 'r', encoding='utf-8-sig') as f:
            self.mapping = json.load(f)

        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

            # for fast check or benchmark evaluation with no filtering
            self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            self.nSamples = len(self.filtered_index_list)

        if self.maxnum is not None:
            self.nSamples = self.maxnum

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            if self.read_img:
                img_key = 'image-%09d'.encode() % index
                imgbuf = txn.get(img_key)

                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                if self.rgb:
                    img = Image.open(buf).convert('RGB')  # for color image
                else:
                    img = Image.open(buf).convert('L')

            else:
                name_key = 'name-%09d'.encode() % index
                img = txn.get(name_key).decode('utf-8')

        label = self.mapping[label]
        return (img, label)


class ResizeCollate(object):

    def __init__(self, short_size=256, crop_size=227, augment=False):
        self.short_size = short_size
        self.crop_size = crop_size
        self.augment = augment

    def _train_preprocessing(self, img):
        transforms = T.Compose([
            T.Resize(self.short_size),
            T.RandomResizedCrop(self.crop_size),
            T.RandomHorizontalFlip(),
            T.RandomGrayscale(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transforms(img)

    def _valid_preprocessing(self, img):
        transforms = T.Compose([
            T.Resize((self.short_size, self.short_size)),
            T.CenterCrop(self.crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transforms(img)

    def __call__(self, batch):
        images, labels = zip(*batch)

        if self.augment:
            image_tensors = [self._train_preprocessing(x) for x in images]
        else:
            image_tensors = [self._valid_preprocessing(x) for x in images]
        image_tensors = torch.stack(image_tensors)
        labels = torch.as_tensor(labels)
        return image_tensors, labels


class ResizeCollateDemo(ResizeCollate):

    def __init__(self, short_size=256, crop_size=227):
        self.short_size = short_size
        self.crop_size = crop_size

    def __call__(self, batch):
        images, path = zip(*batch)

        image_tensors = [self._valid_preprocessing(x) for x in images]
        image_tensors = torch.stack(image_tensors)
        return image_tensors, path


class ResizeCollateImgPyramid(ResizeCollate):

    def __init__(self, short_size=256, crop_size=227, scales=[]):
        self.short_size = short_size
        self.crop_size = crop_size
        self.scales = scales

    def _preprocessing(self, img, scale=1):
        transforms = T.Compose([
            T.Resize((int(self.short_size * scale), int(self.short_size * scale))),
            T.CenterCrop(int(self.crop_size * scale)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transforms(img)

    def __call__(self, batch):
        images, path = zip(*batch)
        image_pyramids = []
        for scale in self.scales:
            image_tensors = [self._preprocessing(x, scale) for x in images]
            image_tensors = torch.stack(image_tensors)
            image_pyramids.append(image_tensors)
        return image_pyramids, path


class ResizeCollateV2(object):
    """From https://www.kaggle.com/c/landmark-retrieval-2021/discussion/277099
    """

    def __init__(self, crop_size=256, augment=False):
        self.crop_size = crop_size
        self.augment = augment
        assert resizeCollate_v2_OK

    def _train_preprocessing(self, img):
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(quality_lower=99, quality_upper=100),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
            A.Resize(self.crop_size, self.crop_size),
            A.CoarseDropout(max_height=int(self.crop_size * 0.4), max_width=int(self.crop_size * 0.4), max_holes=1, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        return transforms(image=img)['image']

    def _valid_preprocessing(self, img):
        transforms = A.Compose(
            [A.Resize(self.crop_size, self.crop_size),
             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             ToTensorV2()])
        return transforms(image=img)['image']

    def __call__(self, batch):
        images, labels = zip(*batch)

        if self.augment:
            image_tensors = [self._train_preprocessing(np.array(x)) for x in images]
        else:
            image_tensors = [self._valid_preprocessing(np.array(x)) for x in images]
        image_tensors = torch.stack(image_tensors)
        labels = torch.as_tensor(labels)
        return image_tensors, labels


class ResizeCollateDemoV2(ResizeCollate):

    def __init__(self, short_size=256, crop_size=227):
        self.short_size = short_size
        self.crop_size = crop_size

    def __call__(self, batch):
        images, path = zip(*batch)

        image_tensors = [self._valid_preprocessing(np.array(x)) for x in images]
        image_tensors = torch.stack(image_tensors)
        return image_tensors, path
