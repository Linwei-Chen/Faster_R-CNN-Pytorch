from model import ResNet18, ResNet50, ResNet101, FasterRCNN, Pooler
import torch
from lr_scheduler import WarmUpMultiStepLR
import time
from torch import optim
from model import get_default_model

def train():
    from dataset import get_voc_data_set, VOC_CLASSES
    from config import config_parser, ANCHOR_RATIOS, ANCHOR_SIZES, RPN_PRE_NMS_TOP_N, RPN_POST_NMS_TOP_N, \
        ANCHOR_SMOOTH_L1_LOSS_BETA, PROPOSAL_SMOOTH_L1_LOSS_BETA, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, STEP_LR_SIZES, \
        STEP_LR_GAMMA, WARM_UP_FACTOR, WARM_UP_NUM_ITERS
    from utils import device_init
    args = config_parser()
    device = device_init(args)
    model = get_default_model(True)
    data_set = get_voc_data_set()
    optimizer = optim.SGD(model.parameters(),
                          lr=LEARNING_RATE,
                          momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    scheduler = WarmUpMultiStepLR(optimizer,
                                  milestones=STEP_LR_SIZES,
                                  gamma=STEP_LR_GAMMA,
                                  factor=WARM_UP_FACTOR,
                                  num_iters=WARM_UP_NUM_ITERS)

    step = 0
    model.to(device)
    model.train()
    step = model.load(optimizer, scheduler)
    t1 = time.perf_counter()
    for _, (imgs, targets) in enumerate(data_set):
        step += 1
        batch_size = imgs.shape[0]
        image_batch = imgs
        bboxes_batch = torch.cat([i[..., :4] for i in targets], 0).unsqueeze(0)
        labels_batch = torch.cat([i[..., 4] for i in targets], 0).unsqueeze(0).long()
        image_batch, bboxes_batch, labels_batch = \
            image_batch.to(device), bboxes_batch.to(device), labels_batch.to(device)
        # print(f'image_batch.shape:{image_batch.shape}')
        # print(f'bboxes_batch.shape:{bboxes_batch.shape}')
        # print(f'bboxes_batch:{bboxes_batch}')
        # print(f'labels_batch.shape:{labels_batch.shape}')
        # print(f'labels_batch:{labels_batch}')
        anchor_objectness_losses, anchor_transformer_losses, proposal_class_losses, proposal_transformer_losses = \
            model.train().forward(image_batch, bboxes_batch, labels_batch)
        anchor_objectness_loss = anchor_objectness_losses.mean()
        anchor_transformer_loss = anchor_transformer_losses.mean()
        proposal_class_loss = proposal_class_losses.mean()
        proposal_transformer_loss = proposal_transformer_losses.mean()
        loss = anchor_objectness_loss + anchor_transformer_loss + proposal_class_loss + proposal_transformer_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        t2 = time.perf_counter()
        print('step:{} | loss:{:.8f} | time:{:.4f}'.format(step, loss.item(), t2 - t1))
        t1 = time.perf_counter()
        if step != 0 and step % args.save_step == 0:
            model.save(step, optimizer, scheduler)


if __name__ == '__main__':
    while True:
        train()
