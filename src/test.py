import os
import os.path as osp
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image


def draw_box(img_np, boxes_np, tags_np, scores_np=None, relative_coord=False, save_path=None):
    if scores_np is None:
        scores_np = [1.0 for i in tags_np]
    # img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    h, w, _ = img_np.shape
    if relative_coord:
        boxes_np = np.array([
            boxes_np[:, 0] * w,
            boxes_np[:, 1] * h,
            boxes_np[:, 2] * w,
            boxes_np[:, 3] * h,
        ]).T
    plt.figure(figsize=(10, 10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    currentAxis = plt.gca()
    for box, tag, score in zip(boxes_np, tags_np, scores_np):
        from src.dataset import VOC_CLASSES as LABLES
        tag = int(tag)
        label_name = LABLES[tag]
        display_txt = '%s: %.2f' % (label_name, score)
        coords = (box[0], box[1]), box[2] - box[0] + 1, box[3] - box[1] + 1
        color = colors[tag]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(box[0], box[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
    plt.imshow(img_np)
    if save_path is not None:
        # fig, ax = plt.subplots()
        fig = plt.gcf()
        fig.savefig(save_path)
        plt.cla()
        plt.clf()
        plt.close('all')
    else:
        plt.show()


def test(img_path):
    def img_to_tensor_batch(img_path):
        img = Image.open(img_path)
        img_tensor = transforms.ToTensor()(img).unsqueeze(0)
        # print(f'img_tensor:{img_tensor.shape}')
        # print(f'img_tensor:{img_tensor}')
        return img_tensor, img

    from src.model import get_default_model
    from src.utils import device_init, nms
    from src.config import config_parser
    device = device_init(config_parser())
    model = get_default_model(pre_trained=True)
    model.load()
    model = model.to(device)
    img_tensor, img = img_to_tensor_batch(img_path)
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        (detection_bboxes, detection_classes,
         detection_probs, detection_batch_indices) = model.eval().forward(img_tensor)
        # print('bboxes.shape:{}\nclasses.shape:{}\nprobs.shape;{}\nbatch_indices.shape:{}'.format(
        #     detection_bboxes.shape, detection_classes.shape, detection_probs.shape, detection_batch_indices.shape))

        kept_indices = (detection_probs > 0.6).nonzero().view(-1)
        detection_bboxes = detection_bboxes[kept_indices]
        detection_classes = detection_classes[kept_indices]
        detection_probs = detection_probs[kept_indices]
        detection_batch_indices = detection_batch_indices[kept_indices]

        if detection_bboxes.shape[0] != 0:
            # NMS
            kept_indices, _ = nms(detection_bboxes, detection_probs, overlap=0.3, top_k=None)
            detection_bboxes = detection_bboxes[kept_indices]
            detection_classes = detection_classes[kept_indices]
            detection_probs = detection_probs[kept_indices]
            detection_batch_indices = detection_batch_indices[kept_indices]

        # print('bboxes.shape:{}\nclasses.shape:{}\nprobs.shape;{}\nbatch_indices.shape:{}'.format(
        #     detection_bboxes.shape, detection_classes.shape, detection_probs.shape, detection_batch_indices.shape))
        save_dir, img_name = osp.split(img_path)
        save_dir = osp.join(save_dir, 'result')
        from src.utils import create_dir
        create_dir(save_dir)
        draw_box(img_np=np.array(img),
                 boxes_np=detection_bboxes.cpu().numpy(),
                 tags_np=detection_classes.cpu().numpy(),
                 scores_np=detection_probs.cpu().numpy(),
                 save_path=osp.join(save_dir, img_name))


if __name__ == '__main__':
    test_img_dir = '../test_img'
    for root, dirs, files in os.walk(test_img_dir, topdown=True):
        if test_img_dir == root:
            print(root, dirs, files)
            files = [i for i in files if any([j in i for j in ['jpg', 'png', 'jpeg', 'gif', 'tiff']])]
            with open(osp.join(test_img_dir, 'tested.txt'), 'r') as txt:
                txt = txt.readlines()
                txt = [i.strip() for i in txt]
                print(txt)
                files = [i for i in files if i not in txt]
            for file in files:
                file_path = os.path.join(root, file)
                print(f'*** testing:{file_path}')
                test(file_path)
                with open(osp.join(test_img_dir, 'tested.txt'), 'a') as txt:
                    txt.write(file+'\n')
