#!/usr/bin/env python
import argparse
import pytorch_lightning as pl
import warnings
from copy import deepcopy
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from metric.model import BaselineModel
from metric.dataset import ResizeCollate, LmdbDataset, RawDatasetTraining
from metric.sampler import ImbalancedDatasetSampler, MPerClassSampler


def train(opt):
    if opt.manualSeed:
        pl.seed_everything(opt.manualSeed)

    loss_config = dict()
    # Arcface
    loss_config['scale'] = opt.scale
    loss_config['margin'] = opt.margin

    # for DDP
    loss_config['is_ddp'] = True if opt.num_gpus > 1 else False

    if opt.saved_model:
        tags = ['finetune']
    else:
        tags = ['train']

    print('preparing dataset')
    if opt.train_csv_path and opt.valid_csv_path:
        train_dataset = RawDatasetTraining(opt.train_data, label_csv_path=opt.train_csv_path, label_mapping=opt.label_mapping)
        valid_dataset = RawDatasetTraining(opt.valid_data, label_csv_path=opt.valid_csv_path, label_mapping=opt.label_mapping)
    else:
        train_dataset = LmdbDataset(opt.train_data, label_mapping=opt.label_mapping)
        valid_dataset = LmdbDataset(opt.valid_data, label_mapping=opt.label_mapping)

    iter_per_epoch = int(len(train_dataset) / opt.batch_size * opt.data_proportion / opt.num_gpus)
    print('Iteration per epoch: ', iter_per_epoch)

    print('determine the number of training classes')
    if opt.train_csv_path and opt.valid_csv_path:
        labels = deepcopy(train_dataset.label_list)
        labels.extend(deepcopy(valid_dataset.label_list))
    else:
        labels = [x for _, x in LmdbDataset(opt.train_data, read_img=False, label_mapping=opt.label_mapping)]
        labels.extend([x for _, x in LmdbDataset(opt.valid_data, read_img=False, label_mapping=opt.label_mapping)])
    nclass = len(set(labels))
    print('Number of training classes: {}'.format(nclass))

    print('Crop size: {}'.format(opt.crop_size))
    short_size = int(opt.crop_size * 1.1)
    print('Short size: {}'.format(short_size))

    if opt.weight_path:
        print('Using balanced data sampler')
        if opt.train_csv_path and opt.valid_csv_path:
            dataset = RawDatasetTraining(opt.train_data, label_csv_path=opt.train_csv_path, read_img=False, label_mapping=opt.label_mapping)
        else:
            dataset = LmdbDataset(opt.train_data, read_img=False, label_mapping=opt.label_mapping)
        sampler = ImbalancedDatasetSampler(dataset, opt.weight_path)
        train_loader = DataLoader(train_dataset,
                                  sampler=sampler,
                                  num_workers=opt.workers,
                                  batch_size=opt.batch_size,
                                  collate_fn=ResizeCollate(augment=True, crop_size=opt.crop_size, short_size=short_size),
                                  pin_memory=True,
                                  drop_last=True)
    elif opt.m_per_class:
        print('Using m per class sampler with m = {}'.format(opt.m_per_class))
        if opt.train_csv_path and opt.valid_csv_path:
            dataset = RawDatasetTraining(opt.train_data, label_csv_path=opt.train_csv_path, read_img=False, label_mapping=opt.label_mapping)
        else:
            dataset = LmdbDataset(opt.train_data, read_img=False, label_mapping=opt.label_mapping)
        sampler = MPerClassSampler(dataset, opt.m_per_class, length_before_new_iter=len(train_dataset))
        train_loader = DataLoader(train_dataset,
                                  sampler=sampler,
                                  num_workers=opt.workers,
                                  batch_size=opt.batch_size,
                                  collate_fn=ResizeCollate(augment=True, crop_size=opt.crop_size, short_size=short_size),
                                  pin_memory=True,
                                  drop_last=True)
    else:
        train_loader = DataLoader(train_dataset,
                                  num_workers=opt.workers,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  collate_fn=ResizeCollate(augment=True, crop_size=opt.crop_size, short_size=short_size),
                                  pin_memory=True,
                                  drop_last=True)

    valid_loader = DataLoader(valid_dataset,
                              num_workers=opt.workers,
                              batch_size=opt.batch_size,
                              shuffle=False,
                              collate_fn=ResizeCollate(augment=False, crop_size=opt.crop_size, short_size=short_size),
                              pin_memory=True)

    if opt.saved_model:
        baseline_model = BaselineModel.load_from_checkpoint(opt.saved_model,
                                                            nclasses=nclass,
                                                            finetune=True,
                                                            iter_per_epoch=iter_per_epoch,
                                                            init_lr=opt.init_lr,
                                                            num_epochs=opt.num_epoch,
                                                            embedding_dim=opt.embedding,
                                                            loss_config=loss_config,
                                                            cs=opt.cs,
                                                            global_only=opt.global_only,
                                                            optim=opt.optim,
                                                            map_at_k=opt.map_at_k)
    else:
        baseline_model = BaselineModel(nclasses=nclass,
                                       finetune=False,
                                       iter_per_epoch=iter_per_epoch,
                                       init_lr=opt.init_lr,
                                       num_epochs=opt.num_epoch,
                                       embedding_dim=opt.embedding,
                                       loss_config=loss_config,
                                       cs=opt.cs,
                                       global_only=opt.global_only,
                                       optim=opt.optim,
                                       map_at_k=opt.map_at_k)

    checkpoint = pl.callbacks.ModelCheckpoint(monitor='valid/MAP@R',
                                              mode='max',
                                              verbose=True,
                                              save_weights_only=True,
                                              dirpath=f'./saved_models/{opt.exp_name}/',
                                              filename='highest_recall')
    learning_rate = pl.callbacks.LearningRateMonitor(logging_interval='step')

    logger = pl.loggers.TensorBoardLogger(save_dir=f'./saved_models/{opt.exp_name}/log', default_hp_metric=False)

    precision = 16 if opt.mix else 32
    accumulate_grad_batches = 10 if opt.batch_size < 24 else 1
    wb_logger = WandbLogger(project=opt.project_name, tags=tags, name=opt.exp_name, config=opt)

    if opt.num_gpus == 1:
        trainer = pl.Trainer(logger=[logger, wb_logger],
                             gpus=opt.num_gpus,
                             precision=precision,
                             max_epochs=opt.num_epoch,
                             benchmark=True,
                             val_check_interval=opt.valInterval,
                             limit_val_batches=opt.data_proportion,
                             limit_train_batches=opt.data_proportion,
                             accumulate_grad_batches=accumulate_grad_batches,
                             callbacks=[checkpoint, learning_rate])
    else:
        if opt.weight_path or opt.m_per_class:
            warnings.warn('Customized sampler is not supported in distributed training currently.')
        find_unused_parameters = True if opt.global_only else False
        trainer = pl.Trainer(logger=[logger, wb_logger],
                             replace_sampler_ddp=True,
                             gpus=opt.num_gpus,
                             accelerator='ddp',
                             plugins=DDPPlugin(find_unused_parameters=find_unused_parameters),
                             precision=precision,
                             benchmark=True,
                             max_epochs=opt.num_epoch,
                             val_check_interval=opt.valInterval,
                             limit_val_batches=opt.data_proportion,
                             limit_train_batches=opt.data_proportion,
                             accumulate_grad_batches=accumulate_grad_batches,
                             callbacks=[checkpoint, learning_rate])

    if opt.saved_model:
        trainer.validate(baseline_model, valid_loader, ckpt_path=None)

    trainer.fit(baseline_model, train_loader, valid_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', required=True, help='Project name')
    parser.add_argument('--exp_name', required=True, help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--train_csv_path', help='path to training dataset (for input raw img)')
    parser.add_argument('--valid_csv_path', help='path to validation dataset (for input raw img)')
    parser.add_argument('--label_mapping', default='data/mapping.json', help='Data mapping json location')
    parser.add_argument('--weight_path', help='path of the weighting json')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of GPU use')
    parser.add_argument('--init_lr', type=float, default=1e-6, help='initial learning rate')
    parser.add_argument('--manualSeed', type=int, help='for random seed setting')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--valInterval', type=float, default=1., help='Interval between each validation')
    parser.add_argument('--num_epoch', type=int, default=40, help='number of epoch to train for')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--embedding', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--m_per_class', type=int, help='Select m sample per class')
    parser.add_argument('--data_proportion', type=float, default=1.0, help='Training/validating on x propotion of data')
    parser.add_argument('--crop_size', type=int, default=227, help='Image crop size')
    parser.add_argument('--scale', type=float, default=220, help='Arcface scale')
    parser.add_argument('--margin', type=float, default=18.6, help='Arcface margin')
    parser.add_argument('--cs', type=int, default=1024, help='Cs in DOLG')
    parser.add_argument('--mix', action='store_true', help='Using mix-precision training')
    parser.add_argument('--global_only', action='store_true', help='Training the global feature only')
    parser.add_argument('--optim', default='adamw', help='optimizer to use')
    parser.add_argument('--map_at_k', type=int, default=100, help='MAP@K')
    opt = parser.parse_args()

    train(opt)
