import os
from math import ceil
from glob import glob
from osgeo import gdal
import numpy as np
import cv2
import torch
from torch.nn import functional as F
import albumentations as A


class Augmenation:

    def __init__(self,flag = 0):
        self.flag = flag

    def transform(self, data):
        if self.flag==0:
            return data
        elif self.flag==1:
            return data[:,:,::-1,:]
        elif self.flag==2:
            return data[:,:,:,::-1]
        else:
            return data[:,:,::-1,::-1]

    def inverse_transform(self,data):
        if self.flag == 0:
            return data
        elif self.flag == 1:
            return data[:, :, ::-1, :]
        elif self.flag == 2:
            return data[:, :, :, ::-1]
        else:
            return data[:, :, ::-1, ::-1]


class Sobel:

    def __init__(self,ksize=3):
        self.ksize = ksize
        self._caches = {}

    @staticmethod
    def _generate_sobel_kernel(shape, axis):
        """
        shape must be odd: eg. (5,5)
        axis is the direction, with 0 to positive x and 1 to positive y
        """
        k = np.zeros(shape, dtype=np.float32)
        p = [
            (j, i)
            for j in range(shape[0])
            for i in range(shape[1])
            if not (i == (shape[1] - 1) / 2.0 and j == (shape[0] - 1) / 2.0)
        ]

        for j, i in p:
            j_ = int(j - (shape[0] - 1) / 2.0)
            i_ = int(i - (shape[1] - 1) / 2.0)
            k[j, i] = (i_ if axis == 0 else j_) / float(i_ * i_ + j_ * j_)
        return torch.from_numpy(k).unsqueeze(0)

    @classmethod
    def kernel(cls, ksize=None):
        sobel_x, sobel_y = (cls._generate_sobel_kernel((ksize, ksize), i) for i in (0, 1))
        sobel_ker = torch.cat([sobel_x, sobel_y], dim=0).view(2, 1, ksize, ksize)
        return sobel_ker.to('cuda')

    def conv2d(self,data,flag=None, mode='reflect'):
        pad_size = self.ksize // 2
        data = F.pad(data, (pad_size, pad_size, pad_size, pad_size), mode=mode)
        data = F.conv2d(data, self.kernel(self.ksize))
        if flag=='xy':
            return data
        else:
            data = torch.sum(data**2,dim=1,keepdim=True)
            return data**0.5


def get_files(FOLDER, extend=r'.tif', train_ratio=0.8, seed=0):
    """
    Walk through dataset folder, extract all the IDs and divide them into train/val/test sets by using adjustable ratio parameter.
    Default:
    0.8 Train
    """

    image_dir = os.path.join(FOLDER, '*{}'.format(extend))
    all_files = glob(image_dir)

    all_names = []
    for file in all_files:
        base_name = os.path.basename(file)
        all_names.append(base_name)

    # np.random.seed(seed)
    # train_names = np.random.choice(all_names, size=round(len(all_names) * train_ratio), replace=False)
    # val_names = np.setdiff1d(all_names, train_names)

    return all_names


def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.2),
        A.OneOf([
            A.MedianBlur(blur_limit=3, p=0.2),
            A.Blur(blur_limit=3, p=0.2),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.2),
    ]
    return A.Compose(train_transform)


def to_tensor(blob):
    if isinstance(blob, np.ndarray):
        return torch.from_numpy(blob)
    if isinstance(blob, int) or isinstance(blob, float):
        return torch.Tensor(blob)

    if isinstance(blob, dict):
        ts = {}
        for k, v in blob.items():
            ts[k] = to_tensor(v)
        return ts

    if isinstance(blob, list):
        ts = list([to_tensor(e) for e in blob])
        return ts
    if isinstance(blob, tuple):
        # namedtuple
        if hasattr(blob, '_fields'):
            ts = {k: to_tensor(getattr(blob, k)) for k in blob._fields}
            ts = type(blob)(**ts)
        else:
            ts = tuple([to_tensor(e) for e in blob])
        return ts


def to_device(blob, device, *args, **kwargs):
    if hasattr(blob, 'to'):
        return blob.to(device, *args, **kwargs)
    if isinstance(blob, torch.Tensor):
        return blob.to(device, *args, **kwargs)

    if isinstance(blob, dict):
        ts = {}
        for k, v in blob.items():
            ts[k] = to_device(v, device)
        return ts

    if isinstance(blob, list):
        ts = list([to_device(e, device) for e in blob])
        return ts
    if isinstance(blob, tuple):
        # namedtuple
        if hasattr(blob, '_fields'):
            ts = {k: to_device(getattr(blob, k), device) for k in blob._fields}
            ts = type(blob)(**ts)
        else:
            ts = tuple([to_device(e, device) for e in blob])
        return ts
    return blob

def mask_to_direction(mask, ksize=9):

    mask = mask.astype(np.uint8)

    pad_width = ksize//2

    mask = np.pad(mask, ((pad_width, pad_width), (pad_width, pad_width)), mode='reflect')

    dist_transform_int = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=0)
    dist_transform_out = cv2.distanceTransform(1 - mask, distanceType=cv2.DIST_L2, maskSize=0)

    dstx_int = cv2.Sobel(dist_transform_int, cv2.CV_64F, 1, 0, ksize=ksize)
    dsty_int = cv2.Sobel(dist_transform_int, cv2.CV_64F, 0, 1, ksize=ksize)

    # angle_int = np.arctan2(dsty,dstx) / np.pi * 180

    dstx_out = cv2.Sobel(dist_transform_out, cv2.CV_64F, 1, 0, ksize=ksize)
    dsty_out = cv2.Sobel(dist_transform_out, cv2.CV_64F, 0, 1, ksize=ksize)

    dstx = dstx_int * mask + dstx_out * (1 - mask)
    dsty = dsty_int * mask + dsty_out * (1 - mask)

    xy = np.stack([dstx, dsty], axis=0)

    xy = F.normalize(torch.from_numpy(xy), dim=0)

    return xy[:,pad_width:-pad_width, pad_width:-pad_width]

def writeImage(in_file,out_file,data):
    '''
    :param im_data: 波段-行-列
    :param im_geotrans: 放射变换六参数
    :param im_proj: 投影
    :param path: 保存文件名
    :return:
    '''
    dataset = gdal.Open(in_file)
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    del dataset

    if 'int' in data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(data.shape) == 3:
        im_bands, im_height, im_width = data.shape
    elif len(data.shape) == 2:
        data = data[None,...]
        im_bands, im_height, im_width = data.shape
    else:
        return -1

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(out_file, int(im_width), int(im_height), int(im_bands), datatype)
    if dataset != None:
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    else:
        return -1
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(data[i])
    del dataset



def predict_full_image(image_path, out_path,call_back):
    batch_size,out_size,inner_size,extend = call_back.params
    image_list = glob(os.path.join(image_path, '*{ext}'.format(ext=extend)))

    for kk,image_file in enumerate(image_list):
        print(kk+1,'===',len(image_list))
        dataset = gdal.Open(image_file)
        if dataset is None:
            return -1

        image = dataset.ReadAsArray()
        image = np.transpose(image,[1,2,0])
        rows,cols,bands = image.shape

        rows_newInnerNum = (rows // inner_size) if (rows % inner_size == 0) else (rows // inner_size + 1)
        cols_newInnerNum = (cols // inner_size) if (cols % inner_size == 0) else (cols // inner_size + 1)

        inner_pad_left_right, inner_pad_top_bottom = (cols_newInnerNum * inner_size - cols), (rows_newInnerNum * inner_size - rows)

        image = np.pad(image, ((0, inner_pad_top_bottom), (0, inner_pad_left_right), (0, 0)), mode='constant')

        pad_size = (out_size - inner_size)//2

        image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size),(0, 0)), mode='constant')

        maskArray = np.zeros([rows_newInnerNum, cols_newInnerNum, out_size, out_size,5], np.float32)

        for i in range(rows_newInnerNum):

            print(i+1, '===>', rows_newInnerNum)

            cube_list = []

            mask_list = []

            for j in range(cols_newInnerNum):

                topleft_rows = i * inner_size

                topleft_cols = j * inner_size

                cube_list.append(
                    image[topleft_rows:topleft_rows + out_size, topleft_cols:topleft_cols + out_size]
                )

            n = int(ceil(len(cube_list) / batch_size))

            for k in range(n):

                if k == n-1:
                    cube_data = cube_list[k * batch_size:]
                else:
                    cube_data = cube_list[k * batch_size:(k + 1) * batch_size]

                cube_data = np.stack(cube_data,axis=0)

                cube_data = cube_data.transpose([0,3,1,2])

                # temp_result = 0
                #
                # for n in range(4):
                #
                #     temp_cube_data = Augmenation(flag=n).transform(cube_data)
                #
                #     temp_cube_data = np.squeeze(call_back(np.ascontiguousarray(temp_cube_data)))
                #
                #     temp_result = Augmenation(flag=n).inverse_transform(temp_cube_data) + temp_result
                #
                # mask_list.append(temp_result/4)
                cube_data = call_back(cube_data)
                mask_list.append(cube_data)

            mask_list = np.concatenate(mask_list, axis=0)

            maskArray[i] = mask_list.transpose([0, 2, 3, 1])

        startIndex, endIndex = (out_size - inner_size) // 2, (out_size - inner_size) // 2 + inner_size

        maskArray = maskArray[:,:,startIndex:endIndex,startIndex:endIndex]

        maskArray = maskArray.transpose([0, 2, 1, 3,4])
        maskArray = np.concatenate(maskArray, axis=0)

        maskArray = maskArray.transpose([1, 2, 0,3])
        maskArray = np.concatenate(maskArray, axis=0)

        maskArray = maskArray.transpose([2, 1, 0])
        maskArray = maskArray[:,0:rows, 0:cols]

        mask_name = 'mask_{name}'.format(name=os.path.basename(image_file))
        mask_file = os.path.join(out_path,mask_name)

        writeImage(image_file,mask_file, maskArray)


def predict_cube(image_path,out_path,call_back):
    file_list = glob(
        os.path.join(image_path, '*{ext}'.format(ext=call_back.params[-1]))
    )
    for image_file in file_list:
        image_dataset = gdal.Open(image_file)
        image = image_dataset.ReadAsArray()
        image = image[None,...]
        out_image = call_back(image)
        out_image = np.concatenate([image[:,0:4],out_image],axis=1)
        mask_name = 'mask_{name}'.format(name=os.path.basename(image_file))
        out_file = os.path.join(out_path,mask_name)

        writeImage(image_file, out_file, np.squeeze(out_image))