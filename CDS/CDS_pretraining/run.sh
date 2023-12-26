#!/bin/bash

python CDS_pretraining.py \
  --data-A '/home/libin/UCDIR-conghuihu/datasets/DomainNet/images/clipart' \
  --data-B '/home/libin/UCDIR-conghuihu/datasets/DomainNet/images/sketch' \
  --gpu_id 3 \
  --clean-model '/home/libin/UCDIR-conghuihu/moco_v2_800ep_pretrain.pth.tar' \
  --exp-dir 'domainnet_clipart-sketch_lr_0.003'