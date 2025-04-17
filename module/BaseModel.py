import torch
from torch import nn
from torch.nn import functional as F
from kornia.filters import spatial_gradient,box_blur

from utils.UtilBase import mask_to_direction


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Channel_Attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class Spatial_Attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_Attention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def stochastic_depth(input: torch.Tensor, p: float,
                     mode: str, training: bool = True):
    if not training or p == 0.0:
        # print(f'not adding stochastic depth of: {p}')
        return input

    survival_rate = 1.0 - p
    if mode == 'row':
        shape = [input.shape[0]] + [1] * (input.ndim - 1)  # just converts BXCXHXW -> [B,1,1,1] list
    elif mode == 'batch':
        shape = [1] * input.ndim

    noise = torch.empty(shape, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    # print(f'added sDepth of: {p}')
    return input * noise


class StochasticDepth(nn.Module):
    '''
    Stochastic Depth module.
    It performs ROW-wise dropping rather than sample-wise.
    mode (str): ``"batch"`` or ``"row"``.
                ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                randomly selected rows from the batch.
    References:
      - https://pytorch.org/vision/stable/_modules/torchvision/ops/stochastic_depth.html#stochastic_depth
    '''

    def __init__(self, p=0.5, mode='row'):
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input):
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self):
        s = f"{self.__class__.__name__}(p={self.p})"
        return


# class Residual_block(nn.Module):
#
#     def __init__(self, ch_in, ch_out,kernel_size=3,stride=1,padding=1):
#         """
#         :param ch_in:
#         :param ch_out:
#         """
#         super(Residual_block, self).__init__()
#         self.conv1 = nn.Conv2d(ch_in, ch_out, (kernel_size, kernel_size), (stride,stride), padding=padding)
#         self.bn1 = nn.BatchNorm2d(ch_out)
#         self.conv2 = nn.Conv2d(ch_out, ch_out, (kernel_size, kernel_size), (stride,stride), padding=padding)
#         self.bn2 = nn.BatchNorm2d(ch_out)
#         self.extra = nn.Sequential()
#         if ch_out != ch_in:
#             # [b, ch_in, h, w] => [b, ch_out, h, w]
#             self.extra = nn.Sequential(
#                 nn.Conv2d(ch_in, ch_out, (1, 1)),
#                 nn.BatchNorm2d(ch_out)
#             )
#         # self.base_block = BasicBlock(ch_out)
#
#     def forward(self, x):
#         """
#
#         :param x: [b, ch, h, w]
#         :return:
#         """
#         out = F.relu(self.bn1(self.conv1(x)), inplace=True)
#
#         # out = self.base_block(out)
#
#         out = self.bn2(self.conv2(out))
#         # short cut.
#         # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
#         # element-wise add:
#         out = self.extra(x) + out
#         out = F.relu(out, inplace=True)
#
#         return out

class MSCA(nn.Module):

    def __init__(self, dim):
        super(MSCA, self).__init__()
        # input
        self.conv55 = nn.Conv2d(dim, dim, 5, padding=2)
        self.spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=6, dilation=3)
        # split into multipats of multiscale attention
        # self.conv17_0 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        # self.conv17_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        #
        # self.conv111_0 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        # self.conv111_1 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        #
        # self.conv211_0 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        # self.conv211_1 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv11 = nn.Conv2d(dim, dim, 1)  # channel mixer
        # self.final_conv11 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        skip = x.clone()

        c55 = self.conv55(x)
        c55 = self.spatial(c55)
        # c17 = self.conv17_0(x)
        # c17 = self.conv17_1(c17)
        # c111 = self.conv111_0(x)
        # c111 = self.conv111_1(c111)
        # c211 = self.conv211_0(x)
        # c211 = self.conv211_1(c211)

        # add = c55 + c17 + c111 + c211

        mixer = self.conv11(c55)

        op = mixer * skip

        return op

class LayerScale(nn.Module):
    '''
    Layer scale module.
    References:
      - https://arxiv.org/abs/2103.17239
    '''
    def __init__(self, inChannels, init_value=1e-2):
        super().__init__()
        self.inChannels = inChannels
        self.init_value = init_value
        self.layer_scale = nn.Parameter(init_value * torch.ones((inChannels)), requires_grad=True)

    def forward(self, x):
        if self.init_value == 0.0:
            return x
        else:
            scale = self.layer_scale.unsqueeze(-1).unsqueeze(-1) # C, -> C,1,1
            return scale * x

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.inChannels}, init_value={self.init_value})'


class Residual_block(nn.Module):

    def __init__(self, ch_in, ch_out,kernel_size=3,stride=1,padding=1,ls_init_val=1e-2, drop_path=0.0):
        """
        :param ch_in:
        :param ch_out:
        """
        super(Residual_block, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, 1, 1)

        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, (1, 1)),
                nn.BatchNorm2d(ch_out)
                    )

    def forward(self, x):
        """

        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)

        # out = self.base_block(out)

        out = self.bn2(self.conv2(out))
        # short cut.
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out
        out = F.relu(out, inplace=True)

        return out


class ResNext(nn.Module):
    def __init__(self, dim):
        super(ResNext, self).__init__()
        self.msca = nn.Sequential(
            MSCA(dim),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        skip = x.clone()
        x = F.relu(self.msca(x) + skip, inplace=True)
        return x


class Encoder_block(nn.Module):

    def __init__(self,ch_in, ch_out, num_res_block,kernel_size=3):

        super(Encoder_block, self).__init__()

        self.first_encoder = Residual_block(ch_in, ch_out, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)

        self.middle_enconder = nn.Sequential(
            *[Residual_block(ch_out, ch_out, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2) for _ in range(num_res_block-1)]
        )

        self.pool_encodedr= nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        encoded = self.first_encoder(x)

        encoded = self.middle_enconder(encoded)

        pool_encodedr = self.pool_encodedr(encoded)

        return encoded, pool_encodedr


class Upsample_block(nn.Module):

    def __init__(self,ch_in:int, ch_out:int,kernel_size:int=1,scale_factor: int = 2,class_num:int = 6):
        super(Upsample_block, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(class_num, 1, (kernel_size, kernel_size), (1, 1), padding=(kernel_size - 1) // 2),
            nn.Sigmoid()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, (kernel_size, kernel_size), (1, 1), padding=(kernel_size - 1) // 2),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear',align_corners=True)
        self.scale_factor = scale_factor

    @staticmethod
    def _sharp_fun(f):
        w1 = f**2
        w2 = (1-f)**2
        return w1 / (w1+w2)

    def forward(self, x):
        # scale = self._sharp_fun(scale)
        # scale = scale.detach().float()
        # scale = self.conv0(scale)
        x = self.conv1(x)
        # x = x * scale + x

        if self.scale_factor>1:
            x = self.upsample(x)

        return x


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_chan, out_chan,3,1,1),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(out_chan,out_chan//8,kernel_size = 1,stride = 1,padding = 0,bias = False)
        self.conv2 = nn.Conv2d(out_chan//8,out_chan,kernel_size = 1,stride = 1,padding = 0,bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fcat):
        feat = self.convblk(fcat)							# (N, C, H, W)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)	# (N, C, 1, 1)
        atten = self.conv1(atten)							# (N, C/4, 1, 1)
        atten = self.relu(atten)
        atten = self.conv2(atten)							# (N, C, 1, 1)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)					# (N, C, H, W)
        feat_out = feat_atten + feat
        return feat_out


class Context_head(nn.Module):
    def __init__(self,ch_in:int=3, ch_out:int=32,kernel_size:int=3,scale_factor: int = 2,img_size:int=512):
        super(Context_head, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv_grad = nn.Sequential(
            nn.Conv2d(ch_out, 1, 1)
        )

    def forward(self, x):

        n,b,h,w = x.shape

        x = self.conv(x)

        grad = torch.squeeze(spatial_gradient(x), dim=1)

        grad = grad.view(n, -1, h, w)

        # feat = torch.cat([grad,feat],dim=1)
        #
        # feat = self.conv(grad)



        # gray_feat = self.conv_grad(feat)
        #
        # gray_feat = torch.squeeze(spatial_gradient(gray_feat))
        #
        # gray_feat = F.normalize(gray_feat)

        return grad


class SelFuseFeature(nn.Module):
    def __init__(self, in_channels, shift_n=4, n_class=4,scale=2):
        super(SelFuseFeature, self).__init__()

        self.shift_n = shift_n
        self.n_class = n_class
        self.scale = scale
        self.fuse_conv = nn.Sequential(nn.Conv2d(2*in_channels, 1, kernel_size=1, padding=0),
                                       nn.BatchNorm2d(1),
                                       nn.Sigmoid()
                                       )

    def forward(self, x, df):
        N, _, H, W = df.shape

        new_h = torch.linspace(0, H-1, H).view(-1, 1).repeat(1, W)
        new_w = torch.linspace(0, W-1, W).repeat(H, 1)
        grid = torch.cat((new_h.unsqueeze(2), new_w.unsqueeze(2)), dim=2)
        grid = grid.unsqueeze(0)

        df = df.permute(0,2,3,1)
        df = df[...,[1,0]]

        grid = grid.expand_as(df).to(x.device, dtype=torch.float)
        grid = grid.detach()
        grid = grid + self.scale * df

        grid[..., 0] = 2 * grid[..., 0] / (H - 1) - 1
        grid[..., 1] = 2 * grid[..., 1] / (W - 1) - 1
        select_x = x.clone()
        for i in range(self.shift_n):
            select_x = F.grid_sample(select_x, grid, mode='bilinear', padding_mode='border', align_corners=True)

        return select_x


class Refine_Block(nn.Module):
    def __init__(self, in_channels, init_filters):
        super(Refine_Block,self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels,init_filters,3,1,1),
            nn.BatchNorm2d(init_filters),
            nn.ReLU(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(init_filters, 2*init_filters, 3, 1, 1),
            nn.BatchNorm2d(2*init_filters),
            nn.ReLU(inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(2*init_filters, 4*init_filters, 3, 1, 1),
            nn.BatchNorm2d(4*init_filters),
            nn.ReLU(inplace=True)
        )

        self.encoder4 = nn.Sequential(
            nn.Conv2d(4 * init_filters, 8 * init_filters, 3, 1, 1),
            nn.BatchNorm2d(8*init_filters),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(8 * init_filters, 4 * init_filters, 3,1,1),
            nn.BatchNorm2d(4 * init_filters),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(4 * init_filters, 2 * init_filters, 3, 1, 1),
            nn.BatchNorm2d(2 * init_filters),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(2 * init_filters, init_filters, 3, 1, 1),
            nn.BatchNorm2d(init_filters),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(init_filters, 2, 1)
        )


    def _cal_direction(self,pro):
        mask = (pro.squeeze(dim=1) > 0.5).detach().cpu().numpy()
        direction = []
        n = mask.shape[0]
        for i in range(n):
            direction.append(
                mask_to_direction(mask[i])
            )
        direction = torch.stack(direction,dim=0).to(pro.device, dtype=torch.float)
        return direction

    def forward(self,img, x):

        dir_feat = self._cal_direction(x)
        xx = torch.cat([img,dir_feat],dim=1)
        ecoder_xx1 = self.encoder1(xx)
        ecoder_xx1_pool = self.Maxpool(ecoder_xx1)

        ecoder_xx2 = self.encoder2(ecoder_xx1_pool)
        ecoder_xx2_pool = self.Maxpool(ecoder_xx2)

        ecoder_xx3 = self.encoder3(ecoder_xx2_pool)
        ecoder_xx3_pool = self.Maxpool(ecoder_xx3)

        ecoder_xx4 = self.encoder4(ecoder_xx3_pool)
        ecoder_xx4_pool = self.Maxpool(ecoder_xx4)

        decoder_xx4 = self.up4(ecoder_xx4_pool) + ecoder_xx3_pool

        decoder_xx3 = self.up3(decoder_xx4) + ecoder_xx2_pool

        decoder_xx2 = self.up2(decoder_xx3) + ecoder_xx1_pool

        decoder_xx1 = self.up1(decoder_xx2)

        return F.normalize(decoder_xx1,dim=1)


class Decoder_Block(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(Decoder_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_filters, (1,1))
        self.norm1 = nn.BatchNorm2d(n_filters)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(n_filters, n_filters, (3, 3), (2, 2), (1, 1), output_padding=1)
        self.norm2 = nn.BatchNorm2d(n_filters)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(n_filters, n_filters, (1, 1))
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class Global_block(nn.Module):

    def __init__(self, in_channels=None, scale_size=2, bn_layer=True):
        super(Global_block, self).__init__()

        self.nonLocal_t = nn.Conv2d(in_channels,in_channels // scale_size, (1, 1))

        self.nonLocal_f = nn.Conv2d(in_channels,in_channels // scale_size, (1, 1))

        self.nonLocal_g = nn.Conv2d(in_channels,in_channels // scale_size, (1, 1))

        if bn_layer:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels // scale_size, in_channels, (1, 1)),
                nn.BatchNorm2d(in_channels),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels // scale_size, in_channels, (1, 1)),
            )

        self.scale_size = scale_size

    def forward(self, x):
        b,c,h, w = x.size()

        t = self.nonLocal_t(x)
        f = self.nonLocal_f(x)
        g = self.nonLocal_g(x)

        t = t.view(b, c//self.scale_size, -1)
        f = f.view(b, c//self.scale_size, -1)
        g = g.view(b, c//self.scale_size, -1)

        relations = F.softmax(torch.matmul(t.permute(0, 2, 1), f), -1)

        y = torch.matmul(relations, g.permute(0, 2, 1))

        y = y.permute(0, 2, 1).contiguous()

        y = self.conv(y.view(b, c//self.scale_size, h, w)) + x

        return y


class CBN_Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(CBN_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (kernel_size, kernel_size), stride=(1, 1), padding=kernel_size//2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):

        dilate1_out = F.relu(self.dilate1(x),inplace=True)

        dilate2_out = F.relu(self.dilate2(dilate1_out), inplace=True)
        dilate3_out = F.relu(self.dilate3(dilate2_out), inplace=True)
        dilate4_out = F.relu(self.dilate4(dilate3_out), inplace=True)
        # dilate5_out = F.relu(self.dilate4(dilate4_out), inplace=True)

        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):

        out1 = self.fc2(self.relu(self.fc1(self.avg_pool(x1))))
        out2 = self.fc2(self.relu(self.fc1(self.avg_pool(x2))))
        out = self.sigmoid(out1 + out2)

        return x1 * out + x2 * (1 - out)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)



class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
