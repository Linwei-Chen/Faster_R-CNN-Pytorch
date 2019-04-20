from typing import Tuple, Type
from enum import Enum
import torchvision
import torch
from torch import nn
from typing import Tuple, List, Optional, Union
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np
from region_proposal_network import RegionProposalNetwork
from utils import BBox, beta_smooth_l1_loss, nms
import os
from utils import save_safely


class Base(object):
    OPTIONS = ['resnet18', 'resnet50', 'resnet101']

    @staticmethod
    def from_name(name: str) -> Type['Base']:
        if name == 'resnet18':
            return ResNet18
        elif name == 'resnet50':
            return ResNet50
        elif name == 'resnet101':
            return ResNet101
        else:
            raise ValueError

    def __init__(self, pretrained: bool):
        super().__init__()
        self._pretrained = pretrained

    def features(self) -> Tuple[nn.Module, nn.Module, int, int]:
        raise NotImplementedError


class ResNet18(Base):

    def __init__(self, pretrained: bool):
        super().__init__(pretrained)

    def features(self) -> Tuple[nn.Module, nn.Module, int, int]:
        resnet18 = torchvision.models.resnet18(pretrained=self._pretrained)

        # list(resnet18.children()) consists of following modules
        #   [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU,
        #   [3] = MaxPool2d, [4] = Sequential(Bottleneck...),
        #   [5] = Sequential(Bottleneck...),
        #   [6] = Sequential(Bottleneck...),
        #   [7] = Sequential(Bottleneck...),
        #   [8] = AvgPool2d, [9] = Linear
        children = list(resnet18.children())
        features = children[:-3]
        num_features_out = 256

        hidden = children[-3]
        num_hidden_out = 512

        for parameters in [feature.parameters() for i, feature in enumerate(features) if i <= 4]:
            for parameter in parameters:
                parameter.requires_grad = False

        features = nn.Sequential(*features)

        return features, hidden, num_features_out, num_hidden_out


class ResNet50(Base):

    def __init__(self, pretrained: bool):
        super().__init__(pretrained)

    def features(self) -> Tuple[nn.Module, nn.Module, int, int]:
        resnet50 = torchvision.models.resnet50(pretrained=self._pretrained)

        # list(resnet50.children()) consists of following modules
        #   [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU,
        #   [3] = MaxPool2d, [4] = Sequential(Bottleneck...),
        #   [5] = Sequential(Bottleneck...),
        #   [6] = Sequential(Bottleneck...),
        #   [7] = Sequential(Bottleneck...),
        #   [8] = AvgPool2d, [9] = Linear
        children = list(resnet50.children())
        features = children[:-3]
        num_features_out = 1024

        hidden = children[-3]
        num_hidden_out = 2048

        for parameters in [feature.parameters() for i, feature in enumerate(features) if i <= 4]:
            for parameter in parameters:
                parameter.requires_grad = False

        features = nn.Sequential(*features)

        return features, hidden, num_features_out, num_hidden_out


class ResNet101(Base):

    def __init__(self, pretrained: bool):
        super().__init__(pretrained)

    def features(self) -> Tuple[nn.Module, nn.Module, int, int]:
        resnet101 = torchvision.models.resnet101(pretrained=self._pretrained)

        # list(resnet101.children()) consists of following modules
        #   [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU,
        #   [3] = MaxPool2d, [4] = Sequential(Bottleneck...),
        #   [5] = Sequential(Bottleneck...),
        #   [6] = Sequential(Bottleneck...),
        #   [7] = Sequential(Bottleneck...),
        #   [8] = AvgPool2d, [9] = Linear
        children = list(resnet101.children())
        # print('children:{}'.format(children))
        features = children[:-3]
        # print('features:{}'.format(features))
        num_features_out = 1024

        hidden = children[-3]
        num_hidden_out = 2048

        # 0~4层停止更新
        for parameters in [feature.parameters() for i, feature in enumerate(features) if i <= 4]:
            for parameter in parameters:
                parameter.requires_grad = False

        features = nn.Sequential(*features)

        return features, hidden, num_features_out, num_hidden_out


class RoiPooling(nn.Module):
    def __init__(self):
        super(RoiPooling, self).__init__()
        pass

    def forward(self):
        pass


class Pooler(object):
    class Mode(Enum):
        POOLING = 'pooling'
        ALIGN = 'align'

    OPTIONS = ['pooling', 'align']

    @staticmethod
    def apply(features: Tensor,
              proposal_bboxes: Tensor,
              proposal_batch_indices: Tensor,
              mode: Mode) -> Tensor:
        _, _, feature_map_height, feature_map_width = features.shape
        scale = 1 / 16
        output_size = (7 * 2, 7 * 2)

        if mode == Pooler.Mode.POOLING:
            pool = []
            for (proposal_bbox, proposal_batch_index) in zip(proposal_bboxes, proposal_batch_indices):
                # [0, feature_map_width)
                start_x = max(min(round(proposal_bbox[0].item() * scale), feature_map_width - 1), 0)
                # (0, feature_map_height]
                start_y = max(min(round(proposal_bbox[1].item() * scale), feature_map_height - 1), 0)
                # [0, feature_map_width)
                end_x = max(min(round(proposal_bbox[2].item() * scale) + 1, feature_map_width), 1)
                # (0, feature_map_height]
                end_y = max(min(round(proposal_bbox[3].item() * scale) + 1, feature_map_height), 1)
                roi_feature_map = features[proposal_batch_index, :, start_y:end_y, start_x:end_x]
                pool.append(F.adaptive_max_pool2d(input=roi_feature_map, output_size=output_size))
            pool = torch.stack(pool, dim=0)
        else:
            raise ValueError

        pool = F.max_pool2d(input=pool, kernel_size=2, stride=2)
        return pool


class FasterRCNN(nn.Module):

    def __init__(self, backbone: Base,
                 num_classes: int,
                 pooler_mode: Pooler.Mode,
                 anchor_ratios: List[Tuple[int, int]],
                 anchor_sizes: List[int],
                 rpn_pre_nms_top_n: int,
                 rpn_post_nms_top_n: int,
                 anchor_smooth_l1_loss_beta: Optional[float] = None,
                 proposal_smooth_l1_loss_beta: Optional[float] = None):
        super().__init__()

        self.features, hidden, num_features_out, num_hidden_out = backbone.features()
        self._bn_modules = nn.ModuleList([it for it in self.features.modules() if isinstance(it, nn.BatchNorm2d)] +
                                         [it for it in hidden.modules() if isinstance(it, nn.BatchNorm2d)])

        # NOTE: It's crucial to freeze batch normalization modules for few batches training,
        # which can be done by following processes
        #       (1) Change mode to `eval`
        #       (2) Disable gradient (we move this process into `forward`)
        # 冻结BatchNorm层的原因：
        # https://www.zhihu.com/question/68164562/answer/260539238
        # https://blog.csdn.net/qq_29007291/article/details/86627048
        for bn_module in self._bn_modules:
            for parameter in bn_module.parameters():
                parameter.requires_grad = False

        self.rpn = RegionProposalNetwork(num_features_out,
                                         anchor_ratios,
                                         anchor_sizes,
                                         rpn_pre_nms_top_n,
                                         rpn_post_nms_top_n,
                                         anchor_smooth_l1_loss_beta)

        self.detection = self.Detection(pooler_mode, hidden, num_hidden_out, num_classes, proposal_smooth_l1_loss_beta)

    def forward(self, image_batch: Tensor,
                gt_bboxes_batch: Tensor = None,
                gt_classes_batch: Tensor = None) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor],
                                                          Tuple[Tensor, Tensor, Tensor, Tensor]]:
        # disable gradient for each forwarding process just in case model was switched to `train` mode at any time
        # https://blog.csdn.net/ccbrid/article/details/80573253
        for bn_module in self._bn_modules:
            bn_module.eval()

        features = self.features(image_batch)

        batch_size, _, image_height, image_width = image_batch.shape
        _, _, features_height, features_width = features.shape

        # anchor_bboxes: tensor[batch, anchor_num, 4]
        anchor_bboxes = self.rpn.generate_anchors(image_width,
                                                  image_height,
                                                  num_x_anchors=features_width,
                                                  num_y_anchors=features_height).to(features).repeat(batch_size, 1, 1)
        # 训练模式：
        if self.training:
            (anchor_objectnesses,
             anchor_transformers,
             anchor_objectness_losses,
             anchor_transformer_losses) = self.rpn.forward(features,
                                                           anchor_bboxes,
                                                           gt_bboxes_batch,
                                                           image_width,
                                                           image_height)
            # proposal_bboxes: tensor[batch_size, proposal_num, 4]
            proposal_bboxes = self.rpn.generate_proposals(anchor_bboxes,
                                                          anchor_objectnesses,
                                                          anchor_transformers,
                                                          image_width,
                                                          image_height).detach()
            # it's necessary to detach `proposal_bboxes` here

            # print(f'proposal_bboxes.shape:{proposal_bboxes.shape}')
            # print(f'proposal_bboxes:{proposal_bboxes}')

            (proposal_classes,
             proposal_transformers,
             proposal_class_losses,
             proposal_transformer_losses) = self.detection.forward(features,
                                                                   proposal_bboxes,
                                                                   gt_classes_batch,
                                                                   gt_bboxes_batch)

            return anchor_objectness_losses, anchor_transformer_losses, proposal_class_losses, proposal_transformer_losses
        else:
            anchor_objectnesses, anchor_transformers = self.rpn.forward(features)
            proposal_bboxes = self.rpn.generate_proposals(anchor_bboxes,
                                                          anchor_objectnesses,
                                                          anchor_transformers,
                                                          image_width,
                                                          image_height)
            proposal_classes, proposal_transformers = self.detection.forward(features, proposal_bboxes)

            (detection_bboxes,
             detection_classes,
             detection_probs,
             detection_batch_indices) = self.detection.generate_detections(proposal_bboxes,
                                                                           proposal_classes,
                                                                           proposal_transformers,
                                                                           image_width,
                                                                           image_height)

            return detection_bboxes, detection_classes, detection_probs, detection_batch_indices

    class Detection(nn.Module):

        def __init__(self, pooler_mode: Pooler.Mode,
                     hidden: nn.Module,
                     num_hidden_out: int,
                     num_classes: int,
                     proposal_smooth_l1_loss_beta: float):
            super().__init__()
            self._pooler_mode = pooler_mode
            self.hidden = hidden
            self.num_classes = num_classes
            self._proposal_class = nn.Linear(num_hidden_out, num_classes)
            self._proposal_transformer = nn.Linear(num_hidden_out, num_classes * 4)
            self._proposal_smooth_l1_loss_beta = proposal_smooth_l1_loss_beta
            self._transformer_normalize_mean = torch.tensor([0., 0., 0., 0.], dtype=torch.float)
            self._transformer_normalize_std = torch.tensor([.1, .1, .2, .2], dtype=torch.float)

        def forward(self, features: Tensor,
                    proposal_bboxes: Tensor,
                    gt_classes_batch: Optional[Tensor] = None,
                    gt_bboxes_batch: Optional[Tensor] = None) -> Union[Tuple[Tensor, Tensor],
                                                                       Tuple[Tensor, Tensor, Tensor, Tensor]]:
            batch_size = features.shape[0]

            # 测试模式：
            if not self.training:
                # >>> x = torch.tensor([1, 2, 3])
                # >>> x.repeat(4, 2)
                # tensor([[ 1,  2,  3,  1,  2,  3],
                #         [ 1,  2,  3,  1,  2,  3],
                #         [ 1,  2,  3,  1,  2,  3],
                #         [ 1,  2,  3,  1,  2,  3]])
                # >>> x.repeat(4, 2, 1).size()
                # torch.Size([4, 2, 3])
                proposal_batch_indices = torch.arange(end=batch_size,
                                                      dtype=torch.long,
                                                      device=proposal_bboxes.device) \
                    .view(-1, 1).repeat(1, proposal_bboxes.shape[1])

                pool = Pooler.apply(features,
                                    proposal_bboxes.view(-1, 4),
                                    proposal_batch_indices.view(-1),
                                    mode=self._pooler_mode)
                hidden = self.hidden(pool)
                hidden = F.adaptive_max_pool2d(input=hidden, output_size=1)
                hidden = hidden.view(hidden.shape[0], -1)

                proposal_classes = self._proposal_class(hidden)
                proposal_transformers = self._proposal_transformer(hidden)

                proposal_classes = proposal_classes.view(batch_size, -1, proposal_classes.shape[-1])
                proposal_transformers = proposal_transformers.view(batch_size, -1, proposal_transformers.shape[-1])
                # proposal_classes: tensor[batch_size, ., 1]
                # proposal_bboxes: tensor[batch_size, ., 4]
                return proposal_classes, proposal_transformers
            # 训练模式
            else:
                # find labels for each `proposal_bboxes`
                labels = torch.full((batch_size,
                                     proposal_bboxes.shape[1]),
                                    -1,
                                    dtype=torch.long,
                                    device=proposal_bboxes.device)
                ious = BBox.iou(proposal_bboxes, gt_bboxes_batch)
                proposal_max_ious, proposal_assignments = ious.max(dim=2)
                labels[proposal_max_ious < 0.5] = 0
                fg_masks = proposal_max_ious >= 0.5
                if len(fg_masks.nonzero()) > 0:
                    labels[fg_masks] = gt_classes_batch[fg_masks.nonzero()[:, 0], proposal_assignments[fg_masks]]

                # select 128 x `batch_size` samples
                fg_indices = (labels > 0).nonzero()
                bg_indices = (labels == 0).nonzero()
                fg_indices = fg_indices[torch.randperm(len(fg_indices))[:min(len(fg_indices), 32 * batch_size)]]
                bg_indices = bg_indices[torch.randperm(len(bg_indices))[:128 * batch_size - len(fg_indices)]]
                selected_indices = torch.cat([fg_indices, bg_indices], dim=0)
                selected_indices = selected_indices[torch.randperm(len(selected_indices))].unbind(dim=1)

                proposal_bboxes = proposal_bboxes[selected_indices]
                gt_bboxes = gt_bboxes_batch[selected_indices[0], proposal_assignments[selected_indices]]
                gt_proposal_classes = labels[selected_indices]
                gt_proposal_transformers = BBox.calc_transformer(proposal_bboxes, gt_bboxes)
                batch_indices = selected_indices[0]

                pool = Pooler.apply(features,
                                    proposal_bboxes,
                                    proposal_batch_indices=batch_indices,
                                    mode=self._pooler_mode)
                hidden = self.hidden(pool)
                hidden = F.adaptive_max_pool2d(input=hidden, output_size=1)
                hidden = hidden.view(hidden.shape[0], -1)

                proposal_classes = self._proposal_class(hidden)
                proposal_transformers = self._proposal_transformer(hidden)
                proposal_class_losses, proposal_transformer_losses = self.loss(proposal_classes,
                                                                               proposal_transformers,
                                                                               gt_proposal_classes,
                                                                               gt_proposal_transformers,
                                                                               batch_size,
                                                                               batch_indices)

                return proposal_classes, proposal_transformers, proposal_class_losses, proposal_transformer_losses

        def loss(self, proposal_classes: Tensor,
                 proposal_transformers: Tensor,
                 gt_proposal_classes: Tensor,
                 gt_proposal_transformers: Tensor,
                 batch_size, batch_indices) -> Tuple[Tensor, Tensor]:

            # proposal_transformers: tensor[num_classes, 4]
            proposal_transformers = proposal_transformers.view(-1, self.num_classes, 4)[
                torch.arange(end=len(proposal_transformers), dtype=torch.long), gt_proposal_classes]
            # self._transformer_normalize_mean = torch.tensor([0., 0., 0., 0.], dtype=torch.float)
            # self._transformer_normalize_std = torch.tensor([.1, .1, .2, .2], dtype=torch.float)
            transformer_normalize_mean = self._transformer_normalize_mean.to(device=gt_proposal_transformers.device)
            transformer_normalize_std = self._transformer_normalize_std.to(device=gt_proposal_transformers.device)
            gt_proposal_transformers = \
                (gt_proposal_transformers - transformer_normalize_mean) / transformer_normalize_std
            # scale up target to make regressor easier to learn

            cross_entropies = torch.empty(batch_size, dtype=torch.float, device=proposal_classes.device)
            smooth_l1_losses = torch.empty(batch_size, dtype=torch.float, device=proposal_transformers.device)

            for batch_index in range(batch_size):
                selected_indices = (batch_indices == batch_index).nonzero().view(-1)

                cross_entropy = F.cross_entropy(input=proposal_classes[selected_indices],
                                                target=gt_proposal_classes[selected_indices])

                fg_indices = gt_proposal_classes[selected_indices].nonzero().view(-1)
                smooth_l1_loss = beta_smooth_l1_loss(input=proposal_transformers[selected_indices][fg_indices],
                                                     target=gt_proposal_transformers[selected_indices][fg_indices],
                                                     beta=self._proposal_smooth_l1_loss_beta)

                cross_entropies[batch_index] = cross_entropy
                smooth_l1_losses[batch_index] = smooth_l1_loss

            return cross_entropies, smooth_l1_losses

        def generate_detections(self, proposal_bboxes: Tensor,
                                proposal_classes: Tensor,
                                proposal_transformers: Tensor,
                                image_width: int,
                                image_height: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            batch_size = proposal_bboxes.shape[0]

            proposal_transformers = proposal_transformers.view(batch_size, -1, self.num_classes, 4)
            transformer_normalize_std = self._transformer_normalize_std.to(device=proposal_transformers.device)
            transformer_normalize_mean = self._transformer_normalize_mean.to(device=proposal_transformers.device)
            proposal_transformers = proposal_transformers * transformer_normalize_std + transformer_normalize_mean

            proposal_bboxes = proposal_bboxes.unsqueeze(dim=2).repeat(1, 1, self.num_classes, 1)
            detection_bboxes = BBox.apply_transformer(proposal_bboxes, proposal_transformers)
            detection_bboxes = BBox.clip(detection_bboxes, left=0, top=0, right=image_width, bottom=image_height)
            detection_probs = F.softmax(proposal_classes, dim=-1)

            all_detection_bboxes = []
            all_detection_classes = []
            all_detection_probs = []
            all_detection_batch_indices = []

            for batch_index in range(batch_size):
                for c in range(1, self.num_classes):
                    class_bboxes = detection_bboxes[batch_index, :, c, :]
                    class_probs = detection_probs[batch_index, :, c]
                    threshold = 0.7
                    kept_indices, _ = nms(class_bboxes, class_probs, threshold)
                    class_bboxes = class_bboxes[kept_indices]
                    class_probs = class_probs[kept_indices]

                    all_detection_bboxes.append(class_bboxes)
                    all_detection_classes.append(torch.full((len(kept_indices),), c, dtype=torch.int))
                    all_detection_probs.append(class_probs)
                    all_detection_batch_indices.append(torch.full((len(kept_indices),), batch_index, dtype=torch.long))

            all_detection_bboxes = torch.cat(all_detection_bboxes, dim=0)
            all_detection_classes = torch.cat(all_detection_classes, dim=0)
            all_detection_probs = torch.cat(all_detection_probs, dim=0)
            all_detection_batch_indices = torch.cat(all_detection_batch_indices, dim=0)
            return all_detection_bboxes, all_detection_classes, all_detection_probs, all_detection_batch_indices

    def save(self,
             step: int,
             optimizer,
             scheduler,
             path_to_checkpoints_dir: str = None) -> str:

        from config import config_parser
        args = config_parser()
        if path_to_checkpoints_dir is None:
            path_to_checkpoints_dir = args.save_folder
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir, f'model-{step}.pkl')
        checkpoint = {
            'state_dict': self.state_dict(),
            'step': step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        dir_path, file_name = os.path.split(path_to_checkpoint)
        # save_safely(file=checkpoint, dir_path=dir_path, file_name=file_name)
        save_safely(file=checkpoint, dir_path=dir_path, file_name=f'{args.base_net}_model.pkl')
        print(f'*** save the {step} checkpoint successfully! ')
        return path_to_checkpoint

    def load(self,
             optimizer=None,
             scheduler=None,
             path_to_checkpoint: str = None, ) -> 'Model':
        print('*** loading saved model...')
        from config import config_parser
        args = config_parser()
        if path_to_checkpoint is None:
            path_to_checkpoint = os.path.join(args.save_folder, f'{args.base_net}_model.pkl')
        try:
            checkpoint = torch.load(path_to_checkpoint, map_location='cpu')
            self.load_state_dict(checkpoint['state_dict'])
            step = checkpoint['step']
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print('*** loading saved model successfully! ')
            return step
        except Exception:
            print('*** loading saved model fail! ')
            return 0


def get_default_model(pre_trained=True):
    from dataset import get_voc_data_set, VOC_CLASSES
    from config import ANCHOR_RATIOS, ANCHOR_SIZES, RPN_PRE_NMS_TOP_N, RPN_POST_NMS_TOP_N, \
        ANCHOR_SMOOTH_L1_LOSS_BETA, PROPOSAL_SMOOTH_L1_LOSS_BETA, config_parser
    args = config_parser()
    base_net = {
        'resnet18': ResNet18(pre_trained),
        'resnet50': ResNet50(pre_trained),
        'resnet101': ResNet101(pre_trained)
    }
    default_model = FasterRCNN(
        base_net.get(args.base_net),
        len(VOC_CLASSES),
        pooler_mode=Pooler.Mode.POOLING,
        anchor_ratios=ANCHOR_RATIOS,
        anchor_sizes=ANCHOR_SIZES,
        rpn_pre_nms_top_n=RPN_PRE_NMS_TOP_N,
        rpn_post_nms_top_n=RPN_POST_NMS_TOP_N,
        anchor_smooth_l1_loss_beta=ANCHOR_SMOOTH_L1_LOSS_BETA,
        proposal_smooth_l1_loss_beta=PROPOSAL_SMOOTH_L1_LOSS_BETA
    )
    return default_model


# test
if __name__ == '__main__':
    from config import ANCHOR_RATIOS, ANCHOR_SIZES, RPN_PRE_NMS_TOP_N, RPN_POST_NMS_TOP_N, \
        ANCHOR_SMOOTH_L1_LOSS_BETA, PROPOSAL_SMOOTH_L1_LOSS_BETA, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY
    from torch import optim
    from dataset import get_voc_data_set

    model = get_default_model(True)
    data_set = get_voc_data_set()
    optimizer = optim.SGD(model.parameters(),
                          lr=LEARNING_RATE,
                          momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    model.train()
    for _, (imgs, targets) in enumerate(data_set):
        batch_size = imgs.shape[0]
        image_batch = imgs
        bboxes_batch = torch.cat([i[..., :4] for i in targets], 0).unsqueeze(0)
        labels_batch = torch.cat([i[..., 4] for i in targets], 0).unsqueeze(0).long()
        print(f'image_batch.shape:{image_batch.shape}')
        print(f'bboxes_batch.shape:{bboxes_batch.shape}')
        print(f'bboxes_batch:{bboxes_batch}')
        print(f'labels_batch.shape:{labels_batch.shape}')
        print(f'labels_batch:{labels_batch}')
        anchor_objectness_losses, anchor_transformer_losses, proposal_class_losses, proposal_transformer_losses = \
            model.train().forward(image_batch, bboxes_batch, labels_batch)
        anchor_objectness_loss = anchor_objectness_losses.mean()
        anchor_transformer_loss = anchor_transformer_losses.mean()
        proposal_class_loss = proposal_class_losses.mean()
        proposal_transformer_loss = proposal_transformer_losses.mean()
        loss = anchor_objectness_loss + anchor_transformer_loss + proposal_class_loss + proposal_transformer_loss
        print(f'loss:{loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
