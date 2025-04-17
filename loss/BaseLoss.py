import torch
from torch import nn
from torch.nn import functional as F
from kornia.filters import box_blur

from loss.BaseFunction import LossFunction

class ContourLoss(LossFunction):

    def __init__(self, **kwargs):
        super(ContourLoss, self).__init__(**kwargs)
        self.max_pool = nn.MaxPool2d(kernel_size=5,stride=1,padding=2)

    def contour_loss(self, y_true_mask, y_pred_distance, hard_threshold=0.95):
        loss_list = []
        for y_pred in y_pred_distance:
            y_pred = y_pred.unsqueeze(dim=1)

            int_loss, out_loss = self.within_between_mean(y_pred[:,:,0], y_true_mask)
            bce_loss = self.binary_cross_entropy(y_pred[:,:,0], y_true_mask, hard_threshold=hard_threshold)
            loss_list.append(int_loss + out_loss)
            loss_list.append(bce_loss)

        return loss_list

    def feature_loss(self, y_true_mask, y_pred_mask, hard_threshold=0.95):
        loss_list = []
        for y_pred in y_pred_mask:
            y_pred = y_pred.unsqueeze(dim=1)
            int_loss, out_loss = self.within_between_mean(y_true_mask, y_pred[:,:,1])
            bce_loss = self.binary_cross_entropy(y_pred[:,:,1],y_true_mask,hard_threshold=hard_threshold)
            loss_list.append(int_loss + out_loss)
            loss_list.append(bce_loss)
        return loss_list

    def pool_edge_loss(self,y_true,y_pred):
        edge_pred = y_pred - box_blur(y_pred, kernel_size=(5, 5), border_type='constant')
        y_true = y_true.float()
        edge_true = y_true - box_blur(y_true, kernel_size=(5, 5), border_type='constant')
        edge_loss = self.mse_loss(torch.abs(edge_true), torch.abs(edge_pred))
        return edge_loss[0]

    def scale_loss(self, y_true_mask, y_pred_list):
        ce_loss = []
        jac_loss = []
        for i,y_pred in enumerate(y_pred_list):
            y_true = F.interpolate(y_true_mask,scale_factor=1/(2**(3-i)),recompute_scale_factor=True)
            ce_loss.append(
                self.cross_loss(torch.argmax(y_true, dim=1), y_pred)
            )
            jac_loss.append(
                self.jaccard_function(y_true, torch.softmax(y_pred, dim=1))
            )
        ce_loss = torch.mean(torch.stack(ce_loss))
        jac_loss = torch.mean(torch.stack(jac_loss))
        return [ce_loss,jac_loss]

    def contour_map_loss(self, y_true_mask, y_pred_mask):
        int_loss, out_loss = self.within_between_mean(y_true_mask, y_pred_mask)
        contour_loss = int_loss + out_loss
        bce_loss = self.binary_cross_entropy(y_pred_mask, y_true_mask)
        int_loss, out_loss = self.within_between_mean(y_pred_mask, y_true_mask)
        contour_loss = contour_loss + int_loss + out_loss
        int_loss, out_loss = self.within_between_mean(y_pred_mask, y_pred_mask)
        contour_loss = contour_loss + int_loss + out_loss
        jaccard_loss = self.jaccard_function(y_true_mask,y_pred_mask)
        edge_loss = self.pool_edge_loss(y_true_mask,y_pred_mask)

        return [contour_loss,bce_loss,jaccard_loss,edge_loss]

    def multi_contour_map_loss(self, y_true_mask, y_pred_mask):
        class_num = y_true_mask.shape[1]
        contour_list,bce_list,jaccard_list,edge_list = [],[],[],[]
        for c in range(class_num):
            contour_loss, bce_loss, jaccard_loss, edge_loss = self.contour_map_loss(y_true_mask[:,c:c+1],y_pred_mask[:,c:c+1])
            contour_list.append(contour_loss)
            bce_list.append(bce_loss)
            jaccard_list.append(jaccard_loss)
            edge_list.append(edge_loss)
        contour_loss = torch.mean(torch.stack(contour_list))
        bce_loss = torch.mean(torch.stack(bce_list))
        jaccard_loss = torch.mean(torch.stack(jaccard_list))
        edge_loss= torch.mean(torch.stack(edge_list))
        return [contour_loss, bce_loss, jaccard_loss, edge_loss]

    def true_prob_loss(self,y_true,y_pred):
        y_true = y_true.expand_as(y_pred)
        mask = y_true.ne(y_pred > 0.5)
        loss = F.binary_cross_entropy(y_pred,y_true,reduction='none')
        loss = torch.sum(loss*mask) / torch.sum(mask)
        return [loss]

    def true_map_loss(self,y_true,y_pred):
        # loss = F.mse_loss(y_pred,y_true)
        mask = y_true.ne(y_pred)
        loss = F.smooth_l1_loss(y_pred,y_true,reduction='none')
        loss = torch.sum(loss*mask) / torch.sum(mask+1e-8)
        return [loss]

    def contour_edge_loss(self,y_true_edge, y_pred_mask,kernel_size=5):
        loss = self.edge_function(y_true_edge, y_pred_mask,kernel_size=kernel_size)
        return [loss]

    def consistency_loss(self, y_pred_list):
        loss_list = []
        for pred in y_pred_list:
            int_loss, out_loss = self.within_between_mean(pred[:,0], pred[:,1])
            loss_list.append(int_loss + out_loss)
        return loss_list

    def edge_loss(self,y_true,y_pred,num_classes=2):
        loss_list = []
        # edge_loss = self.edge_function(y_true, pred, kernel_size=5)
        # loss_list.append(edge_loss)
        edge_loss = self.cross_loss(y_true,y_pred,num_classes=num_classes)
        # edge_loss = self.binary_cross_entropy(y_pred, y_true, hard_threshold=0.95)
        loss_list.append(edge_loss)
        return loss_list

    def edge_distance_loss(self,y_true,y_pred,num_classes):
        loss_list = []
        edge_loss = self.cross_loss(y_true,y_pred,num_classes=num_classes)
        if torch.any(torch.isnan(edge_loss)):
            print('edge_distance_loss')
        loss_list.append(edge_loss)
        return loss_list

    def edge_mask_loss(self,y_true_mask,y_pred_mask):
        bce_loss = self.binary_cross_entropy(y_pred_mask, y_true_mask)
        jaccard_loss = self.jaccard_function(y_true_mask, y_pred_mask)
        return [bce_loss,jaccard_loss]

    def angle_loss(self,y_true,y_pred,mask=None):

        y_true = y_true.squeeze(dim=2)
        loss = self.angle_function(y_true, y_pred, mask)
        return [loss]

    def distance_loss(self,y_true,y_pred,num_classes):
        loss = self.cross_loss(y_true,y_pred,num_classes=num_classes)
        if torch.any(torch.isnan(loss)):
            print('distance_loss')
        # y_pred1 = torch.sum(y_pred[:,32::],dim=1,keepdim=True)
        # mask = ((y_true==31) & (y_true==32) & (y_true==33)).detach().float()
        # edge_loss = self.edge_function(mask,y_pred1,kernel_size=3)
        return [loss]

    def mse_loss(self,y_true,y_pred,mask=None):
        # loss = F.smooth_l1_loss(y_pred,y_true,reduction='none')
        loss = torch.abs(y_pred-y_true)
        if isinstance(mask,torch.Tensor):
            loss = torch.sum(loss*mask) / torch.sum(mask)
        else:
            loss = torch.mean(loss)
        return [loss]

