import torch
from loss.BaseLoss import ContourLoss
from utils.UtilRegister import Loss


@Loss.register('MultiLoss')
class MultiTaskLoss(ContourLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, y_pred, **kwargs):
        '''
        :param y_true: 1.y_true_mask ====> 2.y_true_distance  ====> 3.y_true_edge ====> 4.y_true_direction
        :param y_pred: 1.y_pred_mask ====> 2.y_pred_distance  ====> 3.y_pred_direction
        :param sigma: sigma
        :return: total loss
        '''
        self.config.update(kwargs)

        y_true_seg_mask, y_true_edge_mask, y_true_angle = y_pred['y_true_seg_mask'], y_pred['y_true_edge_mask'], y_pred['y_true_angle']

        contour_loss = self.multi_contour_map_loss(y_true_seg_mask, y_pred['contour_map'])

        dir_true_loss = self.true_map_loss(y_true_seg_mask, y_pred['true_map'])

        # scale_loss = self.scale_loss(y_true_seg_mask, y_pred['scale'])

        edge_mask_loss = self.edge_mask_loss(y_true_edge_mask, y_pred['edge_map'])

        angle_loss = self.angle_loss(y_true_angle, y_pred['angle_map'])

        return contour_loss + angle_loss + edge_mask_loss + dir_true_loss


