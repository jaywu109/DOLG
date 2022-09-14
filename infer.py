import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from metric.dataset import ResizeCollateDemo, RawDataset
from metric.model import BaselineModelInfer


def infer(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('preparing dataset')
    test_dataset = RawDataset(opt.test_data, maxnum=opt.maxnum)

    print('Crop size: {}'.format(opt.crop_size))
    short_size = opt.crop_size
    print('Short size: {}'.format(short_size))

    test_loader = DataLoader(test_dataset,
                             num_workers=opt.workers,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             collate_fn=ResizeCollateDemo(crop_size=opt.crop_size, short_size=short_size),
                             pin_memory=True)

    model = BaselineModelInfer.load_from_checkpoint(opt.saved_model,
                                                    strict=False,
                                                    embedding_dim=opt.embedding,
                                                    cs=opt.cs,
                                                    global_only=opt.global_only)

    model.eval().to(device)

    embeddings, attns, img_paths = [], [], []

    with torch.no_grad():
        for img_tensor, img_path in tqdm(test_loader):
            embedding, attn = model(img_tensor.to(device))
            embeddings.append(embedding.cpu().numpy())
            attns.append(attn.cpu().numpy())
            img_paths.extend(img_path)

    embeddings = np.concatenate(embeddings, axis=0)
    attns = np.concatenate(attns, axis=0)
    img_paths = np.array(img_paths)

    data = {'embeddings': embeddings, 'attns': attns, 'paths': img_paths}
    np.savez(opt.output, **data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', required=True, help='root path of the test dataset')
    parser.add_argument('--output', default='./embedding.npz', help='embedding location')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to model for inference")
    parser.add_argument('--embedding', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--crop_size', type=int, default=227, help='Image crop size')
    parser.add_argument('--cs', type=int, default=1024, help='Cs in DELG')
    parser.add_argument('--global_only', action='store_true', help='Training the global feature only')
    parser.add_argument('--maxnum', type=int, help='Maximum number of inference')
    opt = parser.parse_args()

    infer(opt)
