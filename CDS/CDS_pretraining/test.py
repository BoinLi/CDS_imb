import torch
import time
from lib.utils import AverageMeter
import torchvision.transforms as transforms
import numpy as np

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.spatial.distance import cdist

def NN(epoch, net, lemniscate, trainloader, testloader, recompute_memory=0):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    losses = AverageMeter()
    correct = 0.
    total = 0
    testsize = testloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            targets = targets.cuda()
            batchSize = inputs.size(0)
            features = net(inputs)
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.train_labels).cuda()
        trainloader.dataset.transform = transform_bak
    
    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            targets = targets.cuda()
            batchSize = inputs.size(0)
            features = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(1, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)
            yd = yd.narrow(1, 0, 1)

            total += targets.size(0)
            correct += retrieval.eq(targets.data).sum().item()
            
            cls_time.update(time.time() - end)
            end = time.time()

            print('Test [{}/{}]\t'
                  'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  'Top1: {:.2f}'.format(
                  total, testsize, correct*100./total, net_time=net_time, cls_time=cls_time))

    return correct/total

def kNN(epoch, net, lemniscate, trainloader, testloader, K, sigma, recompute_memory=0):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    testsize = testloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.targets).cuda()
    C = trainLabels.max() + 1

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            targets = targets.cuda(async=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.train_labels).cuda()
        trainloader.dataset.transform = transform_bak
    
    top1 = 0.
    top5 = 0.
    end = time.time()
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            end = time.time()
            targets = targets.cuda(async=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1,1))
            cls_time.update(time.time() - end)

            top1 = top1 + correct.narrow(1,0,1).sum().item()
            top5 = top5 + correct.narrow(1,0,5).sum().item()

            total += targets.size(0)

            print('Test [{}/{}]\t'
                  'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  'Top1: {:.2f}  Top5: {:.2f}'.format(
                  total, testsize, top1*100./total, top5*100./total, net_time=net_time, cls_time=cls_time))

    print(top1*100./total)

    return top1/total


def kNN_DA(epoch, net, lemniscate, trainloader, testloader, K, sigma, recompute_memory=0, verbose=False):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    testsize = testloader.dataset.__len__()

    trainsize =  trainloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()

    trainLabels = torch.LongTensor(trainloader.dataset.labels).cuda()
    C = trainLabels.max() + 1


    with torch.no_grad():

        if recompute_memory:
            transform_bak = trainloader.dataset.transform
            trainloader.dataset.transform = testloader.dataset.transform
            temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=4) ## trainloader memory
            for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
                targets = targets.cuda()
                inputs = inputs.cuda()
                batchSize = inputs.size(0)
                features = net(inputs)
                trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.data.t()
            trainLabels = torch.LongTensor(temploader.dataset.labels).cuda()
            trainloader.dataset.transform = transform_bak

        lemniscate.memory = trainFeatures.t()




        top1 = 0.
        top5 = 0.
        end = time.time()
        with torch.no_grad():
            retrieval_one_hot = torch.zeros(K, C).cuda()
            for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
                end = time.time()
                inputs = inputs.cuda()
                targets = targets.cuda()
                batchSize = inputs.size(0)
                features = net(inputs)
                net_time.update(time.time() - end)
                end = time.time()

                dist = torch.mm(features, trainFeatures[:,:trainsize])

                yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
                candidates = trainLabels.view(1, -1).expand(batchSize, -1)
                retrieval = torch.gather(candidates, 1, yi)

                retrieval_one_hot.resize_(batchSize * K, C).zero_()
                retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
                yd_transform = yd.clone().div_(sigma).exp_()
                probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1)), 1)
                _, predictions = probs.sort(1, True)

                # Find which predictions match the target
                correct = predictions.eq(targets.data.view(-1, 1))
                cls_time.update(time.time() - end)

                top1 = top1 + correct.narrow(1, 0, 1).sum().item()
                top5 = top5 + correct.narrow(1, 0, 5).sum().item()

                total += targets.size(0)

                if verbose:
                    print('Test [{}/{}]\t'
                      'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                      'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                      'Top1: {:.2f}  Top5: {:.2f}'.format(
                    total, testsize, top1 * 100. / total, top5 * 100. / total, net_time=net_time, cls_time=cls_time))

        # logging.info(top1 * 100. / total)

    logging.info('Test [{}/{}]\t'
          'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
          'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
          'Top1: {:.2f}  Top5: {:.2f}'.format(
        total, testsize, top1 * 100. / total, top5 * 100. / total, net_time=net_time, cls_time=cls_time))

    return top1 *100./ total



def recompute_memory(epoch, net, lemniscate, trainloader):

    net.eval()
    trainFeatures = lemniscate.memory.t()
    batch_size = 100

    with torch.no_grad():
        transform_bak = trainloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):

            targets = targets.cuda()
            inputs = inputs.cuda()
            c_batch_size = inputs.size(0)
            features = net(inputs)
            features = F.normalize(features)

            trainFeatures[:, batch_idx * batch_size:batch_idx * batch_size + c_batch_size] = features.data.t()


            # if batch_idx * batch_size + c_batch_size > 5000:
            #     break

        trainLabels = torch.LongTensor(temploader.dataset.image_cates_A).cuda()
        trainloader.dataset.transform = transform_bak

    lemniscate.memory = trainFeatures.t()

    lemniscate.memory_first = False



def mAP(features_query, gt_labels_query, features_gallary, gt_labels_gallery):
    mAP_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    #print("unique gt_query",len(mAP_ls))
    scores = - cdist(features_query, features_gallary)#shape 898*893
    #print(scores.shape,scores[0])
    for fi in range(features_query.shape[0]):
        mapi = eval_AP_inner(gt_labels_query[fi], scores[fi], gt_labels_gallery)
        try:
            mAP_ls[gt_labels_query[fi]].append(mapi)
        except:
            print()
    mAP = np.array([np.nanmean(maps) for maps in mAP_ls]).mean()#计算每一个类别平均的平均
    return mAP

def eval_AP_inner(inst_id, scores, gt_labels, top=None):
    pos_flag = gt_labels == inst_id#positive flag list
    tot = scores.shape[0]#total number
    tot_pos = np.sum(pos_flag)#total positive number

    sort_idx = np.argsort(-scores)#index down list according to score
    tp = pos_flag[sort_idx]#score down accuracy
    fp = np.logical_not(tp)#score down accuracy not 

    if top is not None:
        print("top",top)
        top = min(top, tot)
        tp = tp[:top]
        fp = fp[:top]
        tot_pos = min(top, tot_pos)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    try:
        rec = tp / tot_pos
        prec = tp / (tp + fp)
    except:
        print(inst_id, tot_pos)
        return np.nan

    ap = VOCap(rec, prec)
    return ap

def VOCap(rec, prec):
    mrec = np.append(0, rec)
    mrec = np.append(mrec, 1)

    mpre = np.append(0, prec)
    mpre = np.append(mpre, 0)

    for ii in range(len(mpre) - 2, -1, -1):
        mpre[ii] = max(mpre[ii], mpre[ii + 1])

    msk = [i != j for i, j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk] - mrec[0:-1][msk]) * mpre[1:][msk])
    return ap

def retrieval_mAP_cal(features_A, targets_A, features_B, targets_B):

    res_A=mAP(features_A,targets_A,features_B,targets_B)
    res_B=mAP(features_B,targets_B,features_A,targets_A)
    
    return res_A, res_B



def compute_features(eval_loader, net, args):
    print('Computing features...')
    net.eval()

    features_A = torch.zeros(eval_loader.dataset.domainA_size, args.low_dim).cuda()

    targets_all_A = torch.zeros(eval_loader.dataset.domainA_size, dtype=torch.int64).cuda()
    

    for i, (images_A, targets_A, indices_A) in enumerate(eval_loader):
        with torch.no_grad():
            images_A = images_A.cuda(non_blocking=True)

            targets_A = targets_A.cuda(non_blocking=True)

            feats_A = net(images_A)

            features_A[indices_A] = feats_A
            targets_all_A[indices_A] = targets_A
           

    return features_A.cpu().numpy(),  targets_all_A.cpu().numpy()

