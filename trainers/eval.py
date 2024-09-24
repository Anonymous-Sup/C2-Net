import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('..')
from datasets import dataloaders
from tqdm import tqdm


def get_score(acc_list):

    mean = np.mean(acc_list)
    interval = 1.96*np.sqrt(np.var(acc_list)/len(acc_list))

    return mean, interval


def meta_test(data_path, model, way, shot, pre, transform_type,
              query_shot=16, trial=2000):

    eval_loader = dataloaders.meta_test_dataloader(data_path=data_path,
                                                   way=way,
                                                   shot=shot,
                                                   pre=pre,
                                                   transform_type=transform_type,
                                                   query_shot=query_shot,
                                                   trial=trial)
    
    target = torch.LongTensor([i//query_shot for i in range(query_shot*way)]).cuda()

    acc_list = []

    for i, (inp, _) in tqdm(enumerate(eval_loader)):

        inp = inp.cuda()
        max_index = model.meta_test(inp, way=way, shot=shot, query_shot=query_shot)

        acc = 100 * torch.sum(torch.eq(max_index, target)).item() / query_shot / way
        acc_list.append(acc)

    mean, interval = get_score(acc_list)
    return mean, interval


def meta_test_yzw(model, dataloader, way, shot,  trial, query_shot=15, validation=False):
    
    acc_list = []

    if validation==False:
        shot = shot*2
        query_shot = query_shot+1
    
    target = torch.LongTensor([i//query_shot for i in range(query_shot * way)]).cuda()

    for i, (img, vid, target_cam, _, target_view, _) in enumerate(dataloader):
        img = img.cuda()
        # target = vid.cuda()

        max_index = model.meta_test(img, way=way, shot=shot, query_shot=15)

        query_shot = 15
        acc = 100 * torch.sum(torch.eq(max_index, target)).item() / query_shot / way
        acc_list.append(acc)

    if trial > 1:
        mean, interval = get_score(acc_list)
    else:
        mean = np.mean(acc_list)
        interval = 0

    return mean, interval