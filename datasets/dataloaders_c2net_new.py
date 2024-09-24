import os
import math
import torch
import torchvision.datasets as datasets
import numpy as np
from copy import deepcopy
from PIL import Image
from . import samplers, transform_manager
from .samplers import FewshotBatchSampler, RandomSampler, ValSampler
from .bases import ImageDataset
from .skechy_fewshot import Sketchy


import torchvision.transforms as T
from torch.utils.data import DataLoader
from timm.data.random_erasing import RandomErasing

# __factory = {
#     'market1501': Market1501,
#     'dukemtmc': DukeMTMCreID,
#     'msmt17': MSMT17,
#     'occ_duke': OCC_DukeMTMCreID,
#     'veri': VeRi,
#     'VehicleID': VehicleID,
#     'market-sketch': MarketSketch,
#     'sysu_mm01': SYSU_MM01,
#     'sksf_a': SKSF_A,
#     'celebHQ': CELAB_HQR,
#     'sketchy': Sketchy,
#     'tuberlin': TUBerlin,
#     'market-sketch-fewshot': MarketSketch_FewShot,
#     'sksf_a_fewshot': SKSF_A_FewShot,
# }

__factory = {
    'sketchy': Sketchy,
}



def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids, img_paths= zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    # viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids, img_paths

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    # viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


SIZE_TRAIN = [256,128]
SIZE_TEST =  [256,128]
PROB = 0.5
PADDING = 10
PIXEL_MEAN = [0.5, 0.5, 0.5]
PIXEL_STD = [0.5, 0.5, 0.5]
RE_PROB = 0.5

def make_fewshot_dataloader(args, data_root):

    train_transforms = T.Compose([
            T.Resize(SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=PROB),
            T.Pad(PADDING),
            T.RandomCrop(SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD),
            RandomErasing(probability=RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
    ])

    num_workers = 8

    dataset = __factory[args.dataset](root=data_root, NWAY=args.train_way, KSHOT=args.train_shot)
    
    train_set = ImageDataset(dataset.train, train_transforms)

    num_classes = dataset.num_train_pids
    # cam_num = dataset.num_train_cams
    # view_num = dataset.num_train_vids

    meta_train_dataloader = DataLoader(train_set, 
                                       batch_sampler= FewshotBatchSampler(dataset.train, 
                                                      way=args.train_way, shot=args.train_shot, query_shot=args.train_query_shot),
                                        num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=False)


    val_set = ImageDataset(dataset.val, val_transforms)
    val_loader = DataLoader(val_set, 
                            batch_sampler=ValSampler(dataset.val, way=args.train_way, 
                                             shot=args.train_shot, query_shot=args.train_query_shot, trial=args.val_trial),
                            num_workers=num_workers, collate_fn=val_collate_fn
    )

    test_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    test_loader = DataLoader(test_set, 
                              batch_sampler=RandomSampler(dataset.query, dataset.gallery, way=args.test_way, 
                                             shot=args.test_shot, query_shot=args.test_query_shot, trial=args.val_trial),
                              num_workers=num_workers, collate_fn=val_collate_fn
    )

    # gallery_loader = DataLoader(gallery_set, 
    #                             batch_sampler=RandomSampler(dataset.gallery, way=args.test,
    #                                             shot=args.testshot, trial=args.val_trial),
    #                             num_workers=num_workers, collate_fn=val_collate_fn
    # )

    return meta_train_dataloader, val_loader, test_loader, len(dataset.query), num_classes



# def get_dataset(data_path, is_training, transform_type, pre):

#     dataset = __factory
#     dataset = datasets.ImageFolder(
#         data_path,
#         loader=lambda x: image_loader(path=x, is_training=is_training, transform_type=transform_type, pre=pre))

#     return dataset

# def meta_train_dataloader(data_path, way, shots, transform_type):

#     dataset = get_dataset(data_path=data_path,is_training=True,transform_type=transform_type,pre=None)

#     loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_sampler=samplers.meta_batchsampler(data_source=dataset, way=way, shots=shots),
#         num_workers=8,
#         pin_memory=False)

#     return loader



# def meta_test_dataloader(data_path,way,shot,pre,transform_type=None,query_shot=16,trial=1000):

#     dataset = get_dataset(data_path=data_path,is_training=False,transform_type=transform_type,pre=pre)

#     loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_sampler=samplers.random_sampler(data_source=dataset,way=way,shot=shot,query_shot=query_shot,trial=trial),
#         num_workers=8,
#         pin_memory=False)

#     return loader


# def normal_train_dataloader(data_path,batch_size,transform_type):

#     dataset = get_dataset(data_path=data_path,
#                           is_training=True,
#                           transform_type=transform_type,
#                           pre=None)

#     loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=8,
#         pin_memory=False,
#         drop_last=True)

#     return loader


def image_loader(path, is_training, transform_type,pre):

    p = Image.open(path)
    p = p.convert('RGB')

    final_transform = transform_manager.get_transform(is_training=is_training,
                                                      transform_type=transform_type,
                                                      pre=pre)

    p = final_transform(p)

    return p
