from typing import Tuple, List, Optional, Union
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List
import torch.backends.cudnn as cudnn
import os.path as osp
import torch
from torch import Tensor
import os


def save_safely(file, dir_path, file_name):
    """
    save the file safely, if detect the file name conflict,
    save the new file first and remove the old file
    """
    print(dir_path, file_name)
    if not osp.exists(dir_path):
        os.mkdir(dir_path)
        print('*** dir not exist, created one')
    save_path = osp.join(dir_path, file_name)
    if osp.exists(save_path):
        temp_name = save_path + '.temp'
        torch.save(file, temp_name)
        os.remove(save_path)
        os.rename(temp_name, save_path)
        print('*** find the file conflict while saving, saved safely')
    else:
        torch.save(file, save_path)


def update_lr(optimizer, lr):
    """
    update the lr of optimizer
    """
    for param_groups in optimizer.param_groups:
        param_groups['lr'] = lr
    print('*** lr updated:{}'.format(lr))


def beta_smooth_l1_loss(input: Tensor, target: Tensor, beta: float) -> Tensor:
    # return torch.Tensor([1])
    # print(f'input:{input}')
    # print(f'target:{target}')
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    loss = loss.sum() / (input.numel() + 1e-8)
    return loss


class BBox(object):
    def __init__(self, left: float, top: float, right: float, bottom: float):
        super().__init__()
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def __repr__(self) -> str:
        return 'BBox[l={:.1f}, t={:.1f}, r={:.1f}, b={:.1f}]'.format(
            self.left, self.top, self.right, self.bottom)

    def tolist(self) -> List[float]:
        return [self.left, self.top, self.right, self.bottom]

    @staticmethod
    def to_center_base(bboxes: Tensor) -> Tensor:
        return torch.stack([
            (bboxes[..., 0] + bboxes[..., 2]) / 2,
            (bboxes[..., 1] + bboxes[..., 3]) / 2,
            bboxes[..., 2] - bboxes[..., 0],
            bboxes[..., 3] - bboxes[..., 1]
        ], dim=-1)

    @staticmethod
    def from_center_base(center_based_bboxes: Tensor) -> Tensor:
        return torch.stack([
            center_based_bboxes[..., 0] - center_based_bboxes[..., 2] / 2,
            center_based_bboxes[..., 1] - center_based_bboxes[..., 3] / 2,
            center_based_bboxes[..., 0] + center_based_bboxes[..., 2] / 2,
            center_based_bboxes[..., 1] + center_based_bboxes[..., 3] / 2
        ], dim=-1)

    @staticmethod
    # 计算修正值
    def calc_transformer(src_bboxes: Tensor, dst_bboxes: Tensor) -> Tensor:
        center_based_src_bboxes = BBox.to_center_base(src_bboxes)
        center_based_dst_bboxes = BBox.to_center_base(dst_bboxes)
        transformers = torch.stack([
            (center_based_dst_bboxes[..., 0] - center_based_src_bboxes[..., 0]) / center_based_dst_bboxes[..., 2],
            (center_based_dst_bboxes[..., 1] - center_based_src_bboxes[..., 1]) / center_based_dst_bboxes[..., 3],
            torch.log(center_based_dst_bboxes[..., 2] / center_based_src_bboxes[..., 2]),
            torch.log(center_based_dst_bboxes[..., 3] / center_based_src_bboxes[..., 3])
        ], dim=-1)
        return transformers

    @staticmethod
    # 锚框 + 修正 = 回归后的边框
    def apply_transformer(src_bboxes: Tensor, transformers: Tensor) -> Tensor:
        center_based_src_bboxes = BBox.to_center_base(src_bboxes)
        center_based_dst_bboxes = torch.stack([
            transformers[..., 0] * center_based_src_bboxes[..., 2] + center_based_src_bboxes[..., 0],
            transformers[..., 1] * center_based_src_bboxes[..., 3] + center_based_src_bboxes[..., 1],
            torch.exp(transformers[..., 2]) * center_based_src_bboxes[..., 2],
            torch.exp(transformers[..., 3]) * center_based_src_bboxes[..., 3]
        ], dim=-1)
        dst_bboxes = BBox.from_center_base(center_based_dst_bboxes)
        return dst_bboxes

    @staticmethod
    def iou(source: Tensor, other: Tensor) -> Tensor:
        source, other = source.unsqueeze(dim=-2).repeat(1, 1, other.shape[-2], 1), \
                        other.unsqueeze(dim=-3).repeat(1, source.shape[-2], 1, 1)

        source_area = (source[..., 2] - source[..., 0]) * (source[..., 3] - source[..., 1])
        other_area = (other[..., 2] - other[..., 0]) * (other[..., 3] - other[..., 1])

        intersection_left = torch.max(source[..., 0], other[..., 0])
        intersection_top = torch.max(source[..., 1], other[..., 1])
        intersection_right = torch.min(source[..., 2], other[..., 2])
        intersection_bottom = torch.min(source[..., 3], other[..., 3])
        intersection_width = torch.clamp(intersection_right - intersection_left, min=0)
        intersection_height = torch.clamp(intersection_bottom - intersection_top, min=0)
        intersection_area = intersection_width * intersection_height

        return intersection_area / (source_area + other_area - intersection_area)

    @staticmethod
    # 判定框是否未出界
    def inside(bboxes: Tensor, left: float, top: float, right: float, bottom: float) -> Tensor:
        return ((bboxes[..., 0] >= left) * (bboxes[..., 1] >= top) *
                (bboxes[..., 2] <= right) * (bboxes[..., 3] <= bottom))

    @staticmethod
    def clip(bboxes: Tensor, left: float, top: float, right: float, bottom: float) -> Tensor:
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]].clamp(min=left, max=right)
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]].clamp(min=top, max=bottom)
        return bboxes


def nms(boxes, scores, overlap=0.5, top_k=None):
    """
    Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    # boxes = boxes.detach()
    # keep shape [num_prior] type: Long
    keep = scores.new(scores.size(0)).zero_().long()
    # print('keep.shape:{}'.format(keep.shape))
    # tensor.numel()用于计算tensor里面包含元素的总数，i.e. shape[0]*shape[1]...
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # print('x1:{}\ny1:{}\nx2:{}\ny2:{}'.format(x1, y1, x2, y2))
    # area shape[prior_num], 代表每个prior框的面积
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # print(f'idx:{idx}')
    # I = I[v >= 0.01]
    if top_k is not None:
        # indices of the top-k largest vals
        idx = idx[-top_k:]
    # 和boxes同类型的空tensor
    # xx1 = boxes.new()
    # yy1 = boxes.new()
    # xx2 = boxes.new()
    # yy2 = boxes.new()
    # w = boxes.new()
    # h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    # Returns the total number of elements in the input tensor.
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        # torch.index_select(input, dim, index, out=None)
        # 将input里面dim维度上序号为idx的元素放到out里面去
        # >>> x
        # tensor([[1, 2, 3],
        #         [3, 4, 5]])
        # >>> z=torch.index_select(x,0,torch.tensor([1,0]))
        # >>> z
        # tensor([[3, 4, 5],
        #         [1, 2, 3]])
        xx1 = x1[idx]
        # torch.index_select(x1, 0, idx, out=xx1)
        yy1 = y1[idx]
        # torch.index_select(y1, 0, idx, out=yy1)
        xx2 = x2[idx]
        # torch.index_select(x2, 0, idx, out=xx2)
        yy2 = y2[idx]
        # torch.index_select(y2, 0, idx, out=yy2)

        # store element-wise max with next highest score
        # 将除置信度最高的prior框外的所有框进行clip以计算inter大小
        # print(f'xx1.shape:{xx1.shape}')
        xx1 = torch.clamp(xx1, min=float(x1[i]))
        yy1 = torch.clamp(yy1, min=float(y1[i]))
        xx2 = torch.clamp(xx2, max=float(x2[i]))
        yy2 = torch.clamp(yy2, max=float(y2[i]))
        # w.resize_as_(xx2)
        # h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        # torch.le===>less and equal to
        idx = idx[IoU.le(overlap)]
    # print(keep, count)
    # keep 包含置信度从大到小的prior框的indices，count表示数量
    # print('keep.shape:{},count:{}'.format(keep.shape, count))
    return keep, count


def create_dir(dir_path):
    if not osp.exists(dir_path):
        os.mkdir(dir_path)


def device_init(args):
    """
    use for initializing the device to train the model
    """
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        cudnn.benchmark = True
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    return device


# 生成一系列锚框
def generate_anchors(anchor_ratios: List[Tuple],
                     anchor_sizes: List,
                     image_width: int,
                     image_height: int,
                     num_x_anchors: int,
                     num_y_anchors: int) -> Tensor:
    # 在原图上生成均匀的网格，除去最外一圈，生成锚框
    center_ys = np.linspace(start=0, stop=image_height, num=num_y_anchors + 2)[1:-1]
    center_xs = np.linspace(start=0, stop=image_width, num=num_x_anchors + 2)[1:-1]
    # 计算锚框横纵比，将尺寸转换为np格式
    ratios = np.array(anchor_ratios)
    ratios = ratios[:, 0] / ratios[:, 1]
    sizes = np.array(anchor_sizes)

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
def generate_proposals(anchor_bboxes: Tensor,
                       objectnesses: Tensor,
                       transformers: Tensor,
                       pre_nms_top_n,
                       post_nms_top_n,
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
        sorted_bboxes = proposal_bboxes[batch_index][sorted_indices[batch_index]][:pre_nms_top_n]
        sorted_probs = proposal_probs[batch_index][sorted_indices[batch_index]][:pre_nms_top_n]
        # 根据IOU阈值对候选框进行筛选
        threshold = 0.7
        kept_indices = nms(sorted_bboxes, sorted_probs, threshold)
        nms_bboxes = sorted_bboxes[kept_indices][:post_nms_top_n]
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
