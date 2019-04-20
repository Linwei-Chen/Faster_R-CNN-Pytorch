from typing import Tuple, List, Optional, Union

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from utils import nms, BBox, beta_smooth_l1_loss


class RegionProposalNetwork(nn.Module):

    def __init__(self, num_features_out: int,
                 anchor_ratios: List[Tuple[int, int]],
                 anchor_sizes: List[int],
                 pre_nms_top_n: int,
                 post_nms_top_n: int,
                 anchor_smooth_l1_loss_beta: float):
        super().__init__()

        self._features = nn.Sequential(
            nn.Conv2d(in_channels=num_features_out, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # 锚框的比例list
        self._anchor_ratios = anchor_ratios
        # 锚框的尺寸list
        self._anchor_sizes = anchor_sizes
        # 锚框的数量
        num_anchor_ratios = len(self._anchor_ratios)
        num_anchor_sizes = len(self._anchor_sizes)
        num_anchors = num_anchor_ratios * num_anchor_sizes

        # 设定nms前后的候选框数量
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self._anchor_smooth_l1_loss_beta = anchor_smooth_l1_loss_beta

        # rpn输出的置信度和边框回归
        self._anchor_objectness = nn.Conv2d(in_channels=512, out_channels=num_anchors * 2, kernel_size=1)
        self._anchor_transformer = nn.Conv2d(in_channels=512, out_channels=num_anchors * 4, kernel_size=1)

    def forward(self, features: Tensor,
                anchor_bboxes: Optional[Tensor] = None,
                gt_bboxes_batch: Optional[Tensor] = None,
                image_width: Optional[int] = None,
                image_height: Optional[int] = None) -> Union[Tuple[Tensor, Tensor],
                                                             Tuple[Tensor, Tensor, Tensor, Tensor]]:
        batch_size = features.shape[0]

        features = self._features(features)
        anchor_objectnesses = self._anchor_objectness(features)
        anchor_transformers = self._anchor_transformer(features)
        # 背景&非背景置信度：[batch_size, ., 2]
        anchor_objectnesses = anchor_objectnesses.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
        # 边框回归：[batch_size, ., 4]
        anchor_transformers = anchor_transformers.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)

        # 非训练模式
        if not self.training:
            return anchor_objectnesses, anchor_transformers
        # 训练模式
        else:
            # remove cross-boundary
            # NOTE: The length of `inside_indices` is guaranteed to be a multiple of `anchor_bboxes.shape[0]`
            # as each batch in `anchor_bboxes` is the same
            # 剔除出界锚框和相应的边框回归
            inside_indices = BBox.inside(anchor_bboxes,
                                         left=0,
                                         top=0,
                                         right=image_width,
                                         bottom=image_height).nonzero().unbind(dim=1)
            # anchor_bboxes: tensor[batch_size, anchor_num, 4]
            inside_anchor_bboxes = anchor_bboxes[inside_indices].view(batch_size, -1, anchor_bboxes.shape[2])
            inside_anchor_objectnesses = anchor_objectnesses[inside_indices].view(batch_size, -1,
                                                                                  anchor_objectnesses.shape[2])
            inside_anchor_transformers = anchor_transformers[inside_indices].view(batch_size, -1,
                                                                                  anchor_transformers.shape[2])

            # find labels for each `anchor_bboxes`
            # 依据IOU给每一个框贴上标签
            # torch.full(): Returns a tensor of size size filled with fill_value.
            labels = torch.full((batch_size, inside_anchor_bboxes.shape[1]),
                                -1,
                                dtype=torch.long,
                                device=inside_anchor_bboxes.device)
            ious = BBox.iou(inside_anchor_bboxes, gt_bboxes_batch)
            # dim 0, 1, 2 = batch, anchor, gt
            # anchor_max_ious: [batch, max_iou_to_all_gt]
            # anchor_assignments: [batch, gt_index_of_max_iou]
            anchor_max_ious, anchor_assignments = ious.max(dim=2)
            # gt_max_ious: [batch, max_iou_to_all_anchor]
            # gt_assignments: [batch, anchor_index_of_max_iou]
            gt_max_ious, gt_assignments = ious.max(dim=1)
            # nonzero():
            # Returns a tensor containing the indices of all non-zero elements of input.
            # Each row in the result contains the indices of a non-zero element in input.
            # >>> torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
            #                                 [0.0, 0.4, 0.0, 0.0],
            #                                 [0.0, 0.0, 1.2, 0.0],
            #                                 [0.0, 0.0, 0.0,-0.4]]))
            # tensor([[ 0,  0],
            #         [ 1,  1],
            #         [ 2,  2],
            #         [ 3,  3]])
            anchor_additions = ((ious > 0) & (ious == gt_max_ious.unsqueeze(dim=1))).nonzero()[:, :2].unbind(dim=1)
            # torch.unbind(tensor, dim=0) → seq
            # Removes a tensor dimension.
            # Returns a tuple of all slices along a given dimension, already without it.
            # >>> torch.unbind(torch.tensor([[1, 2, 3],
            # >>>                            [4, 5, 6],
            # >>>                            [7, 8, 9]]))
            # (tensor([1, 2, 3]), tensor([4, 5, 6]), tensor([7, 8, 9]))
            # anchor_additions: 真框找到的与其IOU最大的框
            # anchor_max_ious: 与某真框的iou大于threshold的anchor
            # labels: [batch, anchor_num]
            labels[anchor_max_ious < 0.3] = 0
            labels[anchor_additions] = 1
            labels[anchor_max_ious >= 0.7] = 1
            # print(f'labels:{labels}')
            # select 256 x `batch_size` samples
            # fg_indices、bg_indices: [n, 2]
            fg_indices = (labels == 1).nonzero()
            bg_indices = (labels == 0).nonzero()
            # torch.randperm():
            # Returns a random permutation of integers from 0 to n - 1.
            # >>> torch.randperm(4)
            # tensor([2, 1, 0, 3])
            # 每张图随机抽 最大 128 个正样本，再用负样本填充至 256
            fg_indices = fg_indices[torch.randperm(len(fg_indices))[:min(len(fg_indices), 128 * batch_size)]]
            bg_indices = bg_indices[torch.randperm(len(bg_indices))[:256 * batch_size - len(fg_indices)]]
            selected_indices = torch.cat([fg_indices, bg_indices], dim=0)
            selected_indices = selected_indices[torch.randperm(len(selected_indices))].unbind(dim=1)
            # selected_indices = (tensor[256*batch_size,], tensor[256*batch_size,]) 表示batch、anchor_index
            # >>> x
            # tensor([[1, 2, 3],
            #         [3, 2, 1]])
            # >>> x[([0,1],[0,1])]
            # tensor([1, 2])

            # inside_anchor_bboxes: tensor[, 4]
            # inside_anchor_bboxes: 未出界的bboxes
            inside_anchor_bboxes = inside_anchor_bboxes[selected_indices]

            # gt_bboxes: tensor[, 4]
            # 每个anchor分配的真框
            gt_bboxes = gt_bboxes_batch[selected_indices[0], anchor_assignments[selected_indices]]

            # 0 or 1
            # gt_anchor_objectnesses: tensor[1, ]
            gt_anchor_objectnesses = labels[selected_indices]

            # 计算anchor与gt的偏移量
            # gt_anchor_transformers: tensor[, 4]
            gt_anchor_transformers = BBox.calc_transformer(inside_anchor_bboxes, gt_bboxes)
            # print('inside_anchor_bboxes:{}, gt_bboxes:{}'.format(inside_anchor_bboxes, gt_bboxes))
            # print(f'gt_anchor_transformers:{gt_anchor_transformers}')

            # batch_indices: tensor[1,]
            batch_indices = selected_indices[0]

            anchor_objectness_losses, anchor_transformer_losses = self.loss(
                inside_anchor_objectnesses[selected_indices],
                inside_anchor_transformers[selected_indices],
                gt_anchor_objectnesses,
                gt_anchor_transformers,
                batch_size,
                batch_indices)

            return anchor_objectnesses, anchor_transformers, anchor_objectness_losses, anchor_transformer_losses

    def loss(self, anchor_objectnesses: Tensor,
             anchor_transformers: Tensor,
             gt_anchor_objectnesses: Tensor,
             gt_anchor_transformers: Tensor,
             batch_size: int,
             batch_indices: Tensor) -> Tuple[Tensor, Tensor]:

        # 建立两张空表
        cross_entropies = torch.empty(batch_size, dtype=torch.float, device=anchor_objectnesses.device)
        smooth_l1_losses = torch.empty(batch_size, dtype=torch.float, device=anchor_transformers.device)
        # 一张图一张图地计算loss
        for batch_index in range(batch_size):
            # print('batch_indices:{} batch_index:{}'.format(batch_indices, batch_index))
            selected_indices = (batch_indices == batch_index).nonzero().view(-1)
            # print('selected_indices:{}'.format(selected_indices))
            cross_entropy = F.cross_entropy(input=anchor_objectnesses[selected_indices],
                                            target=gt_anchor_objectnesses[selected_indices])
            # print(f'gt_anchor_objectnesses:{gt_anchor_objectnesses}')
            fg_indices = gt_anchor_objectnesses[selected_indices].nonzero().view(-1)
            # print(f'fg_indices:{fg_indices}')
            smooth_l1_loss = beta_smooth_l1_loss(input=anchor_transformers[selected_indices][fg_indices],
                                                 target=gt_anchor_transformers[selected_indices][fg_indices],
                                                 beta=self._anchor_smooth_l1_loss_beta)

            cross_entropies[batch_index] = cross_entropy
            smooth_l1_losses[batch_index] = smooth_l1_loss

        return cross_entropies, smooth_l1_losses

    # 生成一系列锚框：tensor[anchor_num, 4]
    def generate_anchors(self, image_width: int,
                         image_height: int,
                         num_x_anchors: int,
                         num_y_anchors: int) -> Tensor:
        # 在原图上生成均匀的网格，除去最外一圈，生成锚框
        center_ys = np.linspace(start=0, stop=image_height, num=num_y_anchors + 2)[1:-1]
        center_xs = np.linspace(start=0, stop=image_width, num=num_x_anchors + 2)[1:-1]
        # 计算锚框横纵比，将尺寸转换为np格式
        ratios = np.array(self._anchor_ratios)
        ratios = ratios[:, 0] / ratios[:, 1]
        sizes = np.array(self._anchor_sizes)

        # NOTE: it's important to let `center_ys` be the major index
        # (i.e., move horizontally and then vertically) for consistency with 2D convolution
        # giving the string 'ij' returns a meshgrid with matrix indexing,
        # i.e., with shape (#center_ys, #center_xs, #ratios)
        # 'xy'代表的是笛卡尔，'ij'代表的是矩阵。
        # X_N, Y_M, xy->(M,N) ij->(N,M)
        center_ys, center_xs, ratios, sizes = np.meshgrid(center_ys, center_xs, ratios, sizes, indexing='ij')
        # >>> x = torch.tensor([1, 2, 3])
        # >>> y = torch.tensor([4, 5, 6])
        # >>> grid_x, grid_y = torch.meshgrid(x, y)
        # >>> grid_x
        # tensor([[1, 1, 1],
        #         [2, 2, 2],
        #         [3, 3, 3]])
        # >>> grid_y
        # tensor([[4, 5, 6],
        #         [4, 5, 6],
        #         [4, 5, 6]])
        # (grid_x, grid_y) 即坐标

        # 将x、y、w、h 打包成np：[., 4]
        center_ys = center_ys.reshape(-1)
        center_xs = center_xs.reshape(-1)
        ratios = ratios.reshape(-1)
        sizes = sizes.reshape(-1)

        widths = sizes * np.sqrt(1 / ratios)
        heights = sizes * np.sqrt(ratios)

        center_based_anchor_bboxes = np.stack((center_xs, center_ys, widths, heights), axis=1)
        center_based_anchor_bboxes = torch.from_numpy(center_based_anchor_bboxes).float()
        anchor_bboxes = BBox.from_center_base(center_based_anchor_bboxes)
        # tensor[anchor_num, 4]
        return anchor_bboxes

    # 生成Proposal: tensor[batch_size, top_n, 4]
    def generate_proposals(self, anchor_bboxes: Tensor,
                           objectnesses: Tensor,
                           transformers: Tensor,
                           image_width: int,
                           image_height: int) -> Tensor:
        batch_size = anchor_bboxes.shape[0]
        # 锚框 + 修正 + clip= 回归后的合理边框
        # proposal_bboxes：[batches, ., 4]
        proposal_bboxes = BBox.apply_transformer(anchor_bboxes, transformers)
        proposal_bboxes = BBox.clip(proposal_bboxes, left=0, top=0, right=image_width, bottom=image_height)

        # 对含物体的所有置信度进行softmax后降序排列
        # proposal_probs: [batches, ., 1]
        proposal_probs = F.softmax(objectnesses[:, :, 1], dim=-1)
        # 对proposal_probs最后一个维度进行排序
        _, sorted_indices = torch.sort(proposal_probs, dim=-1, descending=True)
        nms_proposal_bboxes_batch = []

        for batch_index in range(batch_size):
            # 选出每张图片的 top_n 个候选框
            sorted_bboxes = proposal_bboxes[batch_index][sorted_indices[batch_index]][:self._pre_nms_top_n]
            sorted_probs = proposal_probs[batch_index][sorted_indices[batch_index]][:self._pre_nms_top_n]
            # 根据IOU阈值对候选框进行筛选
            threshold = 0.7
            kept_indices, _ = nms(sorted_bboxes, sorted_probs, threshold)
            nms_bboxes = sorted_bboxes[kept_indices][:self._post_nms_top_n]
            nms_proposal_bboxes_batch.append(nms_bboxes)

        # 每张图的Proposal数量不一，用torch.zeros() 填充对齐
        max_nms_proposal_bboxes_length = max([len(it) for it in nms_proposal_bboxes_batch])
        padded_proposal_bboxes = []

        for nms_proposal_bboxes in nms_proposal_bboxes_batch:
            padded_proposal_bboxes.append(
                torch.cat([
                    nms_proposal_bboxes,
                    # to(nms_proposal_bboxes) 放在同样的设备中，i.e. cpu or cuda
                    # .to(torch.double)/.to(torch.device('cpu'))/.to(torch.tensor([]))
                    torch.zeros(max_nms_proposal_bboxes_length - len(nms_proposal_bboxes), 4).to(nms_proposal_bboxes)
                ])
            )

        padded_proposal_bboxes = torch.stack(padded_proposal_bboxes, dim=0)
        return padded_proposal_bboxes
