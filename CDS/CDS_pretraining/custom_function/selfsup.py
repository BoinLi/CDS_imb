from __future__ import print_function
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from utils_dh import AverageMeter
from test import compute_features, retrieval_mAP_cal, recompute_memory

from return_dataset import  *
import logging
from tqdm import tqdm

def train_selfsup_only(epoch, args, net, lemniscate_s, lemniscate_t, optimizer, info_save):


    total_time = AverageMeter()

    traindirA = args.data_A   
    traindirB = args.data_B

    train_datasetA = TrainDataset(traindirA, 'RS')
    eval_datasetA = EvalDataset(traindirA, 'RS')

    train_datasetB = TrainDataset(traindirB, 'UT')
    eval_datasetB = EvalDataset(traindirB, 'UT')

    train_loaderA = torch.utils.data.DataLoader(
        train_datasetA, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)

    eval_loaderA = torch.utils.data.DataLoader(
        eval_datasetA, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None)
    
    train_loaderB = torch.utils.data.DataLoader(
        train_datasetB, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)

    eval_loaderB = torch.utils.data.DataLoader(
        eval_datasetB, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None)

    
    # set_model_self(train_loaderA, train_loaderB, eval_loaderA, True)

    cross_entropy_loss = nn.CrossEntropyLoss()

    if lemniscate_s.memory_first:
        recompute_memory(epoch, net, lemniscate_s, train_loaderA)

    if lemniscate_t.memory_first:
        recompute_memory(epoch, net, lemniscate_t, train_loaderB)
        features_A, targets_A = compute_features(eval_loaderA, net, args)
        features_B, targets_B = compute_features(eval_loaderB, net, args)

        res_A, res_B = retrieval_mAP_cal(features_A, targets_A, features_B, targets_B)
        print("First res_A: {}; res_B:{} \n".format(res_A, res_B))
        info_save.write("First res_A: {}; res_B:{} \n".format(res_A, res_B))
        info_save.flush()

        

    net.train()


    scaler = torch.cuda.amp.GradScaler()


    for batch_idx, (inputs, targets, indexes) in enumerate(tqdm(train_loaderA)):

        try:
            inputs2, targets2, indexes2 = target_loader_unl_iter.next()
        except:
            target_loader_unl_iter = iter(train_loaderB)
            inputs2, targets2, indexes2 = target_loader_unl_iter.next()

        inputs, targets, indexes = inputs.cuda(), targets.cuda(), indexes.type(torch.LongTensor).cuda()
        inputs2, targets2, indexes2 = inputs2.cuda(), targets2.cuda(), indexes2.type(torch.LongTensor).cuda()

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            features1 = net(inputs)
            features2 = net(inputs2)

            outputs = lemniscate_s(features1, indexes)
            outputs2 = lemniscate_t(features2, indexes2)

            loss_id = 0

            if args.instance:

                if args.instance:
                    # print("args.instance")
                    source_cross = (cross_entropy_loss(outputs, indexes))
                    target_cross = cross_entropy_loss(outputs2, indexes2)

                    loss_id = (source_cross + target_cross) / 2.0

                total_loss = loss_id

            ###
            if args.s2t or args.t2s:
                # print("args.s2t or args.t2s")
                outputs4 = lemniscate_s(features2, indexes2)
                outputs4 = torch.topk(outputs4, min(args.n_neighbor, train_loaderA.dataset.__len__()), dim=1)[0]
                outputs4 = F.softmax(outputs4*args.temp2, dim=1)
                loss_ent4 = -args.lambda_value * torch.mean(torch.sum(outputs4 * (torch.log(outputs4 + 1e-5)), 1))
                loss_cdm = loss_ent4

                if args.s2t:
                    # print("args.s2t")
                    outputs3 = lemniscate_t(features1, indexes)
                    outputs3 = torch.topk(outputs3,  min(args.n_neighbor, train_loaderB.dataset.__len__()), dim=1)[0]
                    outputs3 = F.softmax(outputs3*args.temp2, dim=1)
                    loss_ent3 = -args.lambda_value * torch.mean(torch.sum(outputs3 * (torch.log(outputs3 + 1e-5)), 1))
                    loss_cdm += loss_ent3
                    cdm_loss = loss_cdm/2.0

                total_loss += loss_cdm

            else:
                loss_cdm=0

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)

        scaler.update()
        optimizer.zero_grad()

        lemniscate_s.update_wegiht(features1.detach(), indexes)
        lemniscate_t.update_wegiht(features2.detach(), indexes2)


    lemniscate_s.memory_first = False
    lemniscate_t.memory_first = False

    # logging.info('source-target')
    features_A, targets_A = compute_features(eval_loaderA, net, args)
    features_B, targets_B = compute_features(eval_loaderB, net, args)

    res_A, res_B = retrieval_mAP_cal(features_A, targets_A, features_B, targets_B)
    
    return res_A, res_B

