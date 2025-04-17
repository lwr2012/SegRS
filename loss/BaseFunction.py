import math
import torch
from torch.nn import functional as F

from utils.UtilAttribute import AttrDict
from loss.angle_loss import EuclideanAngleLossWithOHEM


class LossFunction:

    dir_loss = EuclideanAngleLossWithOHEM()

    def __init__(self, **kwargs):
        self.config = AttrDict(**kwargs)

    @classmethod
    def gradient_function(cls, y_pred):
        '''
        y_pred: tensor of shape (B, C, H, W)
        '''
        # length term
        delta_r = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]  # horizontal gradient (B, C, H-1, W)
        delta_c = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]  # vertical gradient   (B, C, H,   W-1)

        delta_r = delta_r[:, :, 1:, :-2] ** 2  # (B, C, H-2, W-2)
        delta_c = delta_c[:, :, :-2, 1:] ** 2  # (B, C, H-2, W-2)
        delta_pred = torch.abs(delta_r + delta_c)

        epsilon = 1e-8  # where is a parameter to avoid square root is zero in practice.
        len_loss = torch.sqrt(delta_pred + epsilon)
        len_loss = torch.mean(len_loss)
        return len_loss

    @classmethod
    def effective_weight(cls, mask, beta=0.999999):
        one_nums = torch.sum((mask == 1).detach().float())
        zero_nums = torch.sum((mask == 0).detach().float())
        one_nums_weight = (1.0 - beta) / (1 - torch.pow(beta, one_nums))
        zero_nums_weight = (1.0 - beta) / (1 - torch.pow(beta, zero_nums))
        one_nums_weight = one_nums_weight.detach()
        zero_nums_weight = zero_nums_weight.detach()
        one_weight = one_nums_weight / (one_nums_weight + zero_nums_weight)
        zero_weight = zero_nums_weight / (one_nums_weight + zero_nums_weight)
        one_weight = (one_weight * 2).detach()
        zero_weight = (zero_weight * 2).detach()
        return one_weight, zero_weight

    @classmethod
    def heaviside_function(cls, y_pred_distance, epsilon=1e-3):
        # y_pred: without activation(-9...,+9...)
        H = 0.5 * (1 + 2 / math.pi * torch.arctan(y_pred_distance / epsilon))
        return H

    @classmethod
    def dirac_function(cls, y_true_distance, epsilon=1e-3):
        return 1 / (1 + torch.exp(y_true_distance) * epsilon)

    @classmethod
    def get_edge_mask(cls, y_true_distance, min_dis=-5, max_dis=5):
        mask = (y_true_distance <= max_dis) & (y_true_distance >= min_dis)
        return mask.detach().float()

    @classmethod
    def mean_function(cls, I, H, epsilon=1e-7):
        return torch.sum(I * H) / torch.sum(H + epsilon)

    @classmethod
    def regression_function(cls, y_pred_distance, y_true_distance,alpha):
        learn_alpha = torch.exp(-alpha[0])
        mse_loss = F.mse_loss((y_pred_distance - 0.5)*learn_alpha, y_true_distance, reduction='mean')

        y_true_sign = torch.sign(y_true_distance).detach().float()
        mask = ((y_pred_distance*y_true_sign) < 0).detach().float()
        sign_loss = -(y_pred_distance*y_true_sign)*mask
        sign_loss = torch.sum(sign_loss) / torch.sum(mask)

        return mse_loss, sign_loss

    @classmethod
    def binary_weight(cls,y_true):
        den = y_true.sum()  # 0
        b, c, h, w = y_true.shape
        num = h * w * c * b
        alpha = den / num
        weight = torch.zeros_like(y_true)
        weight = torch.fill_(weight, 1-alpha)
        weight[y_true > 0] = alpha
        return weight.cuda()


    @classmethod
    def binary_cross_entropy(cls, y_pred, y_true, hard_threshold=None, is_logits=False):
        y_true = y_true.expand_as(y_pred)
        weight = cls.binary_weight(y_true)
        if is_logits:
            bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true,weight=weight, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(y_pred, y_true,weight=weight, reduction='none')

        if hard_threshold:
            hard_mask = (y_pred < hard_threshold).detach().float()
            bce_loss = torch.sum(hard_mask * bce_loss) / torch.sum(hard_mask + 1e-8)
        else:
            bce_loss = torch.mean(bce_loss)

        return bce_loss


    @classmethod
    def cross_loss(cls, y_true_direction, y_pred_direction, reduction='mean',num_classes=6,ignore_index=-100):

        weight = cls.calc_weights(y_true_direction, num_classes)

        y_true_direction = torch.squeeze(y_true_direction,dim=1)

        loss = F.cross_entropy(y_pred_direction, y_true_direction.long(), reduction=reduction,weight=weight,ignore_index=ignore_index)

        return loss

    @classmethod
    def sharpening(cls,pred, T=2):
        pred_edge = 1 - torch.abs(pred-0.5)*2
        return torch.mean(pred_edge)

    @classmethod
    def within_between_mean(cls, y_true_mask, y_pred_mask):

        # y_true_mask = cls.sharpening(y_true_mask,T=0.5)

        int_mean = cls.mean_function(y_pred_mask, y_true_mask)
        out_mean = cls.mean_function(y_pred_mask, 1 - y_true_mask)

        int_mean = int_mean.expand_as(y_pred_mask)
        out_mean = out_mean.expand_as(y_pred_mask)

        int_loss = F.mse_loss(int_mean, y_pred_mask, reduction='none') * y_true_mask
        out_loss = F.mse_loss(out_mean, y_pred_mask, reduction='none') * (1 - y_true_mask)
        int_loss = torch.mean(int_loss)
        out_loss = torch.mean(out_loss)
        return int_loss, out_loss

    @classmethod
    def jaccard_function(cls, y_true, y_pred, eps=1e-7):
        """Calculate Intersection over Union between ground truth and prediction
        Args:
            y_pred (torch.Tensor): predicted tensor
            y_true (torch.Tensor):  ground truth tensor
            eps (float): epsilon to avoid zero division
        Returns:
            float: loss
        """

        intersection = y_true * y_pred
        union = y_true + y_pred - intersection + eps
        loss = torch.mean(1.0 - intersection / union)
        return loss

    @classmethod
    def dice_function(cls, y_true, y_pred, eps=1e-7):
        intersection = y_true * y_pred
        union = y_true + y_pred + eps
        loss = torch.mean(1.0 - 2 * intersection / union)
        return loss

    @classmethod
    def edge_function(cls, y_true, y_pred, kernel_size=5):
        y_pred_pad = F.pad(y_pred,(kernel_size//2,kernel_size//2,kernel_size//2,kernel_size//2),mode='reflect')
        pool = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=0)
        edge_loss = torch.abs((torch.abs(y_pred - pool(y_pred_pad)) - y_true))
        edge_loss = torch.mean(edge_loss)
        return edge_loss

    @classmethod
    def bmc_loss(cls,pred, target, noise_var=64):
        """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
        Args:
          pred: A float tensor of size [batch, 1].
          target: A float tensor of size [batch, 1].
          noise_var: A float number or tensor.
        Returns:
          loss: A float tensor. Balanced MSE Loss.
        """
        logits = - (pred - target.T).pow(2) / (2 * noise_var)  # logit size: [batch, batch]
        loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))  # contrastive-like loss
        loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable

        return loss

    @classmethod
    def angle_function(cls,y_true,y_pred,mask=None):
        # if ||x||==1, euclidean distance eq cosine similarity
        # The value may overflow, so set extreme range to (-0.999999, 0.999999)
        # angle_loss = F.cosine_similarity(y_true, y_pred,dim=1)*0.999999
        y_true = y_true.squeeze()
        angle_loss = torch.sum(
            y_true * y_pred,
            dim=1,
            keepdim=True
        )

        angle_loss = torch.clip(angle_loss, -1 + 1e-6, 1 - 1e-6)

        angle_loss = torch.acos(angle_loss)

        if isinstance(mask,torch.Tensor):
            angle_loss = torch.sum(angle_loss*mask)/torch.sum(mask+1e-8)
        else:
            angle_loss = torch.mean(angle_loss)
        return angle_loss

    # @classmethod
    # def angle_function(cls, y_true, y_pred, gt=None):
    #     loss = cls.dir_loss(y_pred,y_true,gt)
    #     return loss

    @classmethod
    def calc_weights(cls, label_map, num_classes):
        weights = []
        for i in range(num_classes):
            weights.append((label_map == i).sum().data)
        weights = torch.FloatTensor(weights)
        weights_sum = weights.sum()
        return (1 - weights / weights_sum).cuda()