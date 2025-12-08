import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from osgeo import gdal
from random import choice
from operator import itemgetter

def read_txt(path):
    ims, labels = [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            im, label = line.strip().split()
            ims.append(im)
            labels.append(label)
    return ims,labels


def read_label(filename):
    dataset = gdal.Open(filename)  # 打开文件

    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数

    # im_geotrans = dataset.GetGeoTransform() #仿射矩阵
    # im_proj = dataset.GetProjection() #地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
    # temp = np.zeros((5,im_data.shape[1],im_data.shape[2]))

    del dataset
    return im_data

def np_to_variable(x, is_cuda=True, is_training=False, is_lable=False,dtype=torch.FloatTensor):
    if is_training:
        v = torch.from_numpy(x).type(dtype)
    elif is_lable:
        v = torch.tensor(x).type(dtype)

def data_augmentation(rgb,randN):
    if randN in [0, 1, 2, 3]:
        rgb = np.rot90(rgb, randN, axes=(0, 1))
    if randN == 4:
        rgb = np.rot90(rgb, 3, axes=(0, 1))
        rgb = np.fliplr(rgb)
    if randN == 5:
        rgb = np.rot90(rgb, 3, axes=(0, 1))
        rgb = np.flipud(rgb)
    if randN == 6:
        rgb = np.flipud(rgb)
    if randN == 7:
        rgb = np.fliplr(rgb)
    return rgb


def read_tiff(filename):
    dataset = gdal.Open(filename)  # 打开文件

    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数

    # im_geotrans = dataset.GetGeoTransform() #仿射矩阵
    # im_proj = dataset.GetProjection() #地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
    # temp = np.zeros((5,im_data.shape[1],im_data.shape[2]))
    # 255/最大值
    get_elements = itemgetter(0, 1, 2, 3, 4, 5, 6)
    maxvalue = [3.1921571493148806e-05, 3.417375683784485e-05, 3.89959454536438e-05, 4.6756818890571595e-05, 5.478029847145081e-05, 6.224329471588134e-05, 5.871944427490235e-05, 226.0, 245.0, 230.0, 34.13053431408359]
    #maxvalue = [216.0, 214.0, 216.0, 30.07901954650879, 2268.0, 2587.0, 3339.0, 3964.0, 6104.0, 5867.5, 8277.5]
    #maxvalue = [2268.0, 2587.0, 3339.0, 3964.0, 6104.0, 5867.5, 8277.5, 33.01944351196289] #8band

    for ii in range(len(maxvalue)):
        im_data[ii, ...] = im_data[ii, ...] * 255 / maxvalue[ii]

    del dataset
    return list(get_elements(im_data))#[:7]

def read_image(filename,randN1, device,augmentation=False):
    get_elements = itemgetter(0, 1, 2, 3, 4, 5, 6)
    transform = transforms.Compose(
        [
            transforms.Normalize(
                mean = list(get_elements([0.54171125, 0.48649422, 0.44863061, 0.45274524, 0.44778408, 0.50526287, 0.49983328, 0.31098102, 0.33248206, 0.28747562, -0.64581024])),
            std = list(get_elements([0.04818002, 0.05690863, 0.0744094, 0.08903361, 0.0946957, 0.10636539, 0.09566861, 0.30465061, 0.27811639, 0.2959994, 0.11722139])))#[:7][:7]
            #     mean=[0.39903103, 0.43737722, 0.46030192, -0.31113325, 0.15779845, 0.15015308, 0.17155797, 0.11297999, 0.43602747, 0.27068823, 0.09745705],
            # std = [0.32014577, 0.32815302, 0.3208351, 0.17182075, 0.0619265, 0.0573153, 0.05575203, 0.05458134, 0.11599873, 0.07957992, 0.03835422])
            # mean=[0.15509938, 0.14781779, 0.16948901, 0.11077119, 0.43804327, 0.27022855, 0.09668206, -0.28062309],std=[0.0611714, 0.05668418, 0.05547993, 0.0539083, 0.11600339, 0.07835197, 0.03763121, 0.15423846]#8band
        ]
    )
    # print(filename)
    image = read_tiff(filename)
    image = np.array(image)
    image = np.transpose(image, (1, 2, 0))
    if augmentation == True:
        image = data_augmentation(image, randN1)
        image = np.copy(image)
    image = transforms.ToTensor()(image)

    image = image.to(torch.float32).to(device)
    image = transform(image).to(device)
    return image


# def class_7(filename):
#     label = np.array(read_tiff(filename))
#     label_7 = label
#     for i in range(len(label)):
#         for j in range(len(label[i])):
#             if label[i][j] in range(0,3):
#                 label_7[i][j]=0
#             elif label[i][j] in range(3,7):
#                 label_7[i][j]=1
#             elif label[i][j] in range(7,11):
#                 label_7[i][j]=2
#             elif label[i][j] in range(11,13):
#                 label_7[i][j]=3
#             elif label[i][j] in range(13,16):
#                 label_7[i][j]=4
#             elif label[i][j] in range(16,19):
#                 label_7[i][j]=5
#             elif label[i][j] == 19:
#                 label_7[i][j]=6
#     return label_7


class UnetDataset(Dataset):
    def __init__(self, txtpath,labelreplace,train,device):
        super().__init__()
        self.ims,  self.labels = read_txt(txtpath)
        self.labelreplace = labelreplace
        self.device = device
        self.train = train



    def __getitem__(self, index):
        Set = set([0, 1, 2, 3, 4, 5, 6, 7])
        randN1 = int(choice(list(Set)))
        root_dir = ''
        augmentation = False

        im_path1 = os.path.join(root_dir, self.ims[index])
        im_path =im_path1.replace("image","bigimage")  #coldem11bandgb
        background_path = im_path1.replace("image","bigdatabgimage") #backgroundgb

        label_path1 = os.path.join(root_dir, self.labels[index])
        label_path =label_path1.replace("label", 'label')
        label_path2 =label_path1.replace("label", self.labelreplace)#label5wstructure400mDU


        image =read_image(im_path, randN1,device =self.device,augmentation=False)
        image_background = read_image(background_path,randN1, device =self.device,augmentation=False)


        #labelmulti



        # 20类
        # label = np.asarray(read_label(label_path), dtype=np.int32)
        # if augmentation == True:
        #     label = data_augmentation(label, randN1)
        #     label = np.copy(label)
        # label = torch.from_numpy(label).long().to(self.device)

        label2 = np.asarray(read_label(label_path2), dtype=np.int32)


        if augmentation == True:
            label2 = data_augmentation(label2, randN1)
            label2 = np.copy(label2)

        if np.any(label2==1):
            label_sence = 1
        else:
            label_sence = 0
        label_downsample4 = np.resize(label2, (64 // 4, 64 // 4))
        label_downsample4= torch.from_numpy(label_downsample4).long().to(self.device)
        label2 = torch.from_numpy(label2).long()
        label_sence = torch.from_numpy(np.asarray(label_sence, dtype=np.int32)).long().to(self.device)

        if self.train == True:
            label_flat = label2.view(-1)
            ones_indices = torch.nonzero(label_flat == 1, as_tuple=False).squeeze()
            num_samples = int(ones_indices.numel() * 0.9)  # 选择80%的位置
            random_indices = torch.randperm(ones_indices.numel())[:num_samples]
            selected_indices = ones_indices[random_indices]

            # 创建与 label_flat 大小相同的随机掩码张量
            mask_flat = torch.zeros_like(label_flat)

            # 在选中的位置上将掩码值设置为1.00
            mask_flat[selected_indices] = torch.tensor(1, dtype=torch.long)
            mask_tensor = mask_flat.view(label2.shape).to(self.device)
        else:
            mask_tensor = label2.to(self.device)
        label2 = label2.to(self.device)
        return image, image_background, label_downsample4, label2,label_sence,label_path

    def __len__(self):
        return len(self.ims)