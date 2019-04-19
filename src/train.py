from src.model import ResNet18, ResNet50, ResNet101, FasterRCNN, Pooler
import torch
from src.lr_scheduler import WarmUpMultiStepLR
import time
from torch import optim


def train():
    from src.dataset import get_voc_data_set, VOC_CLASSES
    from src.config import config_parser, ANCHOR_RATIOS, ANCHOR_SIZES, RPN_PRE_NMS_TOP_N, RPN_POST_NMS_TOP_N, \
        ANCHOR_SMOOTH_L1_LOSS_BETA, PROPOSAL_SMOOTH_L1_LOSS_BETA, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, STEP_LR_SIZES, \
        STEP_LR_GAMMA, WARM_UP_FACTOR, WARM_UP_NUM_ITERS
    from src.utils import device_init
    args = config_parser()
    device = device_init(args)
    model = FasterRCNN(
        ResNet18(True),
        len(VOC_CLASSES),
        pooler_mode=Pooler.Mode.POOLING,
        anchor_ratios=ANCHOR_RATIOS,
        anchor_sizes=ANCHOR_SIZES,
        rpn_pre_nms_top_n=RPN_PRE_NMS_TOP_N,
        rpn_post_nms_top_n=RPN_POST_NMS_TOP_N,
        anchor_smooth_l1_loss_beta=ANCHOR_SMOOTH_L1_LOSS_BETA,
        proposal_smooth_l1_loss_beta=PROPOSAL_SMOOTH_L1_LOSS_BETA
    )
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
