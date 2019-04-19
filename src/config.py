import argparse
from typing import Tuple, List
import os

CHANNEL_MEANS = (104, 117, 123)
IMAGE_MIN_SIDE: float = 600.0
IMAGE_MAX_SIDE: float = 1000.0

ANCHOR_RATIOS: List[Tuple[int, int]] = [(1, 2), (1, 1), (2, 1)]
ANCHOR_SIZES: List[int] = [128, 256, 512]

RPN_PRE_NMS_TOP_N: int = 12000
RPN_POST_NMS_TOP_N: int = 2000

ANCHOR_SMOOTH_L1_LOSS_BETA: float = 1.0
PROPOSAL_SMOOTH_L1_LOSS_BETA: float = 1.0

LEARNING_RATE: float = 0.001
MOMENTUM: float = 0.9
WEIGHT_DECAY: float = 0.0005
STEP_LR_SIZES: List[int] = [50000, 70000]
STEP_LR_GAMMA: float = 0.1
WARM_UP_FACTOR: float = 0.3333
WARM_UP_NUM_ITERS: int = 500

NUM_STEPS_TO_DISPLAY: int = 20
NUM_STEPS_TO_SNAPSHOT: int = 10000
NUM_STEPS_TO_FINISH: int = 90000


def config_parser():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
    # train_set = parser.add_mutually_exclusive_group()
    # 训练集与基础网络设定
    parser.add_argument('--voc_data_set_root', default='/Users/chenlinwei/Dataset/VOC0712trainval',
                        help='data_set root directory path')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers used in dataloading')
    # 文件保存路径
    parser.add_argument('--save_folder', default='./saved_model/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save_step', default=100, type=int,
                        help='Directory for saving checkpoint models')
    # 恢复训练
    parser.add_argument('--base_net', default='resnet18', choices=['resnet18', 'resnet50', 'resnet101'],
                        help='Pretrained base model')
    # 优化器参数设置
    # parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
    #                     help='initial learning rate')
    # parser.add_argument('--momentum', default=0.9, type=float,
    #                     help='Momentum value for optim')
    # parser.add_argument('--weight_decay', default=5e-4, type=float,
    #                     help='Weight decay for SGD')
    # parser.add_argument('--gamma', default=0.1, type=float,
    #                     help='Gamma update for SGD')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use cuda or not')

    args = parser.parse_args()
    return args
