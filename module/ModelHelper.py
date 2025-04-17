import torch
from torch import nn
from torch.nn import functional as F
from kornia.filters import spatial_gradient

from module.BaseModel import Residual_block,Encoder_block, Decoder_Block, Upsample_block, Context_head,Dblock,ChannelAttention,SelFuseFeature,Refine_Block

from utils.UtilAttribute import AttrDict
from utils.UtilRegister import Model

class_nums = 6

@Model.register('Model')
class ContourModel(nn.Module):

    def __init__(self, **kwargs):
        super(ContourModel, self).__init__()
        '''
        ch_in: int,
        num_classes: int,
        encoder_filters: list[int],
        block_size: list[int],
        global_num: int,
        '''
        self.config = AttrDict(**kwargs)
        distance_num_classes = self.config.get('distance_num_classes', 64)

        self.encoder_block1 = Encoder_block(
            self.config.get('ch_in', 3),
            self.config.get('encoder_filters')[0],
            self.config.get('block_size')[0],
            kernel_size=self.config.get('kernel_size', 3)
        )
        self.encoder_block2 = Encoder_block(
            self.config.get('encoder_filters')[0],
            self.config.get('encoder_filters')[1],
            self.config.get('block_size')[1],
            kernel_size=self.config.get('kernel_size', 3)
        )
        self.encoder_block3 = Encoder_block(
            self.config.get('encoder_filters')[1],
            self.config.get('encoder_filters')[2],
            self.config.get('block_size')[2],
            kernel_size=self.config.get('kernel_size', 3)
        )
        self.encoder_block4 = Encoder_block(
            self.config.get('encoder_filters')[2],
            self.config.get('encoder_filters')[3],
            self.config.get('block_size')[3],
            kernel_size=self.config.get('kernel_size', 3)
        )

        self.encoder_block5 = Encoder_block(
            self.config.get('encoder_filters')[3],
            self.config.get('encoder_filters')[4],
            self.config.get('block_size')[4],
            kernel_size=self.config.get('kernel_size', 3)
        )

        self.center_NL = Dblock(self.config.get('encoder_filters')[4])

        self.decoder_block1 = Decoder_Block(
            self.config.get('encoder_filters')[4],
            self.config.get('encoder_filters')[4]
        )

        self.decoder_block2 = Decoder_Block(
            self.config.get('encoder_filters')[4],
            self.config.get('encoder_filters')[3]
        )
        self.decoder_block3 = Decoder_Block(
            self.config.get('encoder_filters')[3],
            self.config.get('encoder_filters')[2]
        )
        self.decoder_block4 = Decoder_Block(
            self.config.get('encoder_filters')[2],
            self.config.get('encoder_filters')[1]
        )
        self.decoder_block5 = Decoder_Block(
            self.config.get('encoder_filters')[1],
            self.config.get('encoder_filters')[0]
        )

        self.fusion_1 = ChannelAttention(self.config.get('encoder_filters')[4], 8)
        self.fusion_2 = ChannelAttention(self.config.get('encoder_filters')[3], 8)
        self.fusion_3 = ChannelAttention(self.config.get('encoder_filters')[2], 8)
        self.fusion_4 = ChannelAttention(self.config.get('encoder_filters')[1], 8)
        self.fusion_5 = ChannelAttention(self.config.get('encoder_filters')[0], 8)

        self.upsample_block1 = Upsample_block(
            self.config.get('encoder_filters')[3],
            self.config.get('encoder_filters')[0],
            kernel_size=self.config.get('kernel_size', 3),
            scale_factor=8,
            class_num= class_nums
        )
        self.upsample_block2 = Upsample_block(
            self.config.get('encoder_filters')[2],
            self.config.get('encoder_filters')[0],
            kernel_size=self.config.get('kernel_size', 3),
            scale_factor=4,
            class_num=class_nums
        )
        self.upsample_block3 = Upsample_block(
            self.config.get('encoder_filters')[1],
            self.config.get('encoder_filters')[0],
            kernel_size=self.config.get('kernel_size', 3),
            scale_factor=2,
            class_num=class_nums
        )
        self.upsample_block4 = Upsample_block(
            self.config.get('encoder_filters')[0],
            self.config.get('encoder_filters')[0],
            kernel_size=self.config.get('kernel_size', 3),
            scale_factor=1,
            class_num=class_nums
        )

        self.conv_distance = nn.Sequential(
            nn.Conv2d(
                self.config.get('encoder_filters')[0] * 4,
                distance_num_classes,
                kernel_size=self.config.get('kernel_size', 3),
                padding=self.config.get('kernel_size', 3) // 2
            )
        )

        self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)

        self.conv_scale1 = nn.Sequential(
            nn.Conv2d(self.config.get('encoder_filters')[0],class_nums, 3, 1, 1),
            nn.Softmax()
        )

        self.conv_scale2 = nn.Sequential(
            nn.Conv2d(self.config.get('encoder_filters')[1], class_nums, 3, 1, 1),
            nn.Softmax()
        )

        self.conv_scale3 = nn.Sequential(
            nn.Conv2d(self.config.get('encoder_filters')[2], class_nums, 3, 1, 1),
            nn.Softmax()
        )

        self.conv_scale4 = nn.Sequential(
            nn.Conv2d(self.config.get('encoder_filters')[3], class_nums, 3, 1, 1),
            nn.Softmax()
        )

        self.conv_scale5 = nn.Sequential(
            nn.Conv2d(self.config.get('encoder_filters')[4], class_nums, 3, 1, 1),
            nn.Softmax()
        )


        self.conv_edge = nn.Sequential(
            nn.Conv2d(
                self.config.get('encoder_filters')[0] * 4,
                1,
                kernel_size=self.config.get('kernel_size', 3),
                padding=self.config.get('kernel_size', 3) // 2
            ),
            nn.Sigmoid()
        )

        self.conv_contour = nn.Sequential(
            nn.Conv2d(
                self.config.get('encoder_filters')[0] * 4,
                class_nums,
                kernel_size=self.config.get('kernel_size', 3),
                padding=self.config.get('kernel_size', 3) // 2
            ),
            nn.Softmax()
        )

        self.conv_angle = nn.Sequential(
            nn.Conv2d(self.config.get('encoder_filters')[0] * 4, 1, kernel_size=3,stride=1,padding=1)
        )

        self.dirFuse = SelFuseFeature(1)

        self.sigma = nn.Parameter(
            torch.tensor(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                requires_grad=True,
                dtype=torch.float32
            ).cuda()
        )

        self.multiTaskLoss = self.config.get('callback')
        if hasattr(self.multiTaskLoss,'config'):
            self.multiTaskLoss.config.update(kwargs)
        self.distance_flag = self._set_distance_flag(self.config.get('distance_num_classes'))

    @staticmethod
    def _set_distance_flag(num_classes):
        distance_flag = torch.arange(0,num_classes)
        distance_flag = distance_flag.view(1,-1,1,1)
        return distance_flag.cuda()

    def ont_hot(self, y_true):
        one_hot_true = F.one_hot(y_true.long(), class_nums)
        one_hot_true = torch.permute(one_hot_true.float(), [0, 3, 1, 2])
        return one_hot_true


    def forward(self, x, y_true, **kwargs):

        encoded_1, encoded_pool_1 = self.encoder_block1(x)
        encoded_2, encoded_pool_2 = self.encoder_block2(encoded_pool_1)
        encoded_3, encoded_pool_3 = self.encoder_block3(encoded_pool_2)
        encoded_4, encoded_pool_4 = self.encoder_block4(encoded_pool_3)
        encoded_5, encoded_pool_5 = self.encoder_block5(encoded_pool_4)

        center_block = self.center_NL(encoded_pool_5)

        # decoded_1 = self.decoder_block1(center_block) + encoded_5
        # decoded_2 = self.decoder_block2(decoded_1) + encoded_4
        # decoded_3 = self.decoder_block3(decoded_2) + encoded_3
        # decoded_4 = self.decoder_block4(decoded_3) + encoded_2
        # decoded_5 = self.decoder_block5(decoded_4) + encoded_1

        decoded_1 = self.fusion_1(self.decoder_block1(center_block), encoded_5)
        decoded_2 = self.fusion_2(self.decoder_block2(decoded_1), encoded_4)
        decoded_3 = self.fusion_3(self.decoder_block3(decoded_2), encoded_3)
        decoded_4 = self.fusion_4(self.decoder_block4(decoded_3), encoded_2)
        decoded_5 = self.fusion_5(self.decoder_block5(decoded_4), encoded_1)

        # scale1 = self.conv_scale4(decoded_2)
        # scale2 = self.conv_scale3(decoded_3)
        # scale3 = self.conv_scale2(decoded_4)
        # scale4 = self.conv_scale1(decoded_5)

        out1 = self.upsample_block1(decoded_2)
        out2 = self.upsample_block2(decoded_3)
        out3 = self.upsample_block3(decoded_4)
        out4 = self.upsample_block4(decoded_5)

        out = torch.cat([out1, out2, out3, out4], dim=1)

        contour_map = self.conv_contour(out)

        edge_map = self.conv_edge(out)

        angle_map = self.conv_angle(out)

        angle_map = spatial_gradient(angle_map)

        angle_map = angle_map.squeeze(dim=1)

        angle_map = F.normalize(angle_map)

        if self.training:
            y_true_seg_mask = self.ont_hot(y_true[:, 0])
            y_true_edge_mask = y_true[:, 1:2]
            y_true_angle = y_true[:, 2:]
            # pred_map = self.dirFuse(contour_map, y_true[:,5:])
            true_map = self.dirFuse(y_true_seg_mask, angle_map)

            y_pred = {
                "contour_map": contour_map,
                "y_true_edge_mask": y_true_edge_mask,
                "y_true_seg_mask": y_true_seg_mask,
                "y_true_angle": y_true_angle,
                "y_yrue_flag":y_true[:, 0].long(),
                "true_map": true_map,
                "edge_map": edge_map,
                "angle_map": angle_map
                # "scale": [scale1, scale2, scale3, scale4]
            }

            total_loss = []

            loss_list = self.multiTaskLoss(y_pred)
            for i, loss in enumerate(loss_list):
                precision = torch.exp(-self.sigma[i])
                total_loss.append(
                    torch.sum(precision * loss + self.sigma[i] * self.sigma[i])
                )
            total_loss = torch.sum(
                torch.stack(total_loss, dim=0)
            )
            return total_loss, loss_list

        else:
            final_contour_map = self.dirFuse(contour_map, angle_map)

            return contour_map, final_contour_map, (contour_map + final_contour_map) / 2
