import os
import cv2
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models import unetFEGcn
from models import unet
from models import unetPlusPlus
from models import pspnet
from models import attentionUnet
from models import deeplabv3
from models import swinUnet
from models import bihrnet
from models import res_unet_plus
from models import bisenetv2
from models import FTUNetFormer
from models import res_unet
from utils.mutiunet_dataset_eval import read_tiff,read_image
from osgeo import gdal
from models import unetbrA
from models import unetbrB
from models import unetbrC
from models import unetbrD
from models import unetbx
from models import scnn
from models import scnn2
from models import PACSCNet
from models import logcanplus_model

os.environ['PROJ_LIB'] = r'/home/cv/anaconda3/envs/test/share/proj'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def write_tif(newpath,im_data,im_Geotrans,im_proj, width, height, datatype):
    if len(im_data.shape)==3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    im_width = width
    im_height = height
    im_bands = 1
    diver = gdal.GetDriverByName('GTiff')
    new_dataset = diver.Create(newpath, im_width, im_height, im_bands, datatype)
    new_dataset.SetGeoTransform(im_Geotrans)
    new_dataset.SetProjection(im_proj)
    if im_bands == 1:
        new_dataset.GetRasterBand(1).WriteArray(im_data[0])
    else:
        for i in range(im_bands):
            new_dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del new_dataset


def predict(config,flag,modele):
    device = torch.device('cuda:0')
    selected = modele
    selectedpara = selected + flag
    if selected == 'logcanplus':
        model = logcanplus_model.LoGCANPlus(num_classes=config['num_classes'])
    if selected == 'PACSCNet':
        model = PACSCNet.FFNet()
    if selected  == 'unetFEGcn':
        model = unetFEGcn.UNet(num_classes=config['num_classes'])
    if selected  == "unet":
        model = unet.UNet(num_classes=config['num_classes'])
    if selected == 'unetbx':
        model = unetbx.UNet(num_classes=config['num_classes'])
    if selected == 'unetbrA':
        model = unetbrA.UNet(num_classes=config['num_classes'])
    if selected == 'unetbrB':
        model = unetbrB.UNet(num_classes=config['num_classes'])
    if selected == 'scnn':
        model = scnn.SCNN()
    if selected == 'unetbrC':
        model = unetbrC.UNet(num_classes=config['num_classes'])
    if selected == 'unetbrD':
        model = unetbrD.UNet(num_classes=config['num_classes'])
    if selected == 'unetPlusPlus':
        model = unetPlusPlus.UNetPlusPlus(num_classes=config['num_classes'])
    if selected =="pspnet":
        model = pspnet.PSPNet()
    if selected =="attentionUnet":
        model = attentionUnet.AttentionUnet(num_classes=config['num_classes'])
    if selected =="deeplabv3":
        model = deeplabv3.DeepLab(2)
    if selected =="swinUnet":
        model = swinUnet.SwinTransformerSys()
    if selected =="bihrnet":
        model = bihrnet.HighResolutionNet(num_classes=config['num_classes'])
    if selected =="res_unet_plus":
        model = res_unet_plus.ResUnetPlusPlus(channel=11)
    if selected =="bisenetv2":
        model = bisenetv2.BiSeNetv2()
    if selected =="FTUNetFormer":
        model = FTUNetFormer.FTUNetFormer()
    if selected =="res_unet":
        model = res_unet.ResUnet()
    if selected == 'scnn2':
        model = scnn2.SCNN()
    for i in range(5):
        check_point = os.path.join(config['save_model']['save_path'], selectedpara+'_'+ str(i) +'_jx.pth')

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(check_point, map_location=device), False)
        model.cuda()
        model.eval()

        pre_base_path = os.path.join(config['pre_dir'], 'predict_' + selectedpara+'_test')
        if os.path.exists(pre_base_path) is False:
            os.mkdir(pre_base_path)
        pre_mask_path = os.path.join(pre_base_path, 'mask'+ str(i))
        if os.path.exists(pre_mask_path) is False:
            os.mkdir(pre_mask_path)
        pre_vis_path = os.path.join(pre_base_path, 'vis')
        if os.path.exists(pre_vis_path) is False:
            os.mkdir(pre_vis_path)

        with open(config['img_txt'], 'r', encoding='utf-8') as f:
            for line in f.readlines():
                image_name, _ = line.strip().split()
                root_dir = ''
                image_name1 = os.path.join(root_dir,image_name)
                image_name = image_name1.replace("image", "bigimage")  #coldem11bandgb
                background_path= image_name1.replace("image","bigdatabgimage") #backgroundgb


                #读取坐标体系
                in_ds = gdal.Open(image_name)  # 读取要切的原图
                width = in_ds.RasterXSize  # 获取数据宽度
                height = in_ds.RasterYSize
                im_geotrans = in_ds.GetGeoTransform()  # 获取仿射矩阵信息
                im_proj = in_ds.GetProjection()
                del in_ds
                randN1=0
                image = read_image(image_name, randN1, device,augmentation=False)
                # 加一维,batch_size=1
                image = image.unsqueeze(0)

                image_background = read_image(background_path, randN1, device,augmentation=False)
                image_background = image_background.unsqueeze(0)



                output = model(image,image_background)
                if selected == 'logcanplus':
                    output = output[0]
                # _, pred = output.max(1)
                pred = torch.softmax(output, dim=1)
                pred= torch.round(pred[:, 1, :, :]*100)
                # pred = pred.view(64, 64)
                # mask_im = pred.cpu().numpy().astype(np.uint8)
                mask_im = pred.detach().cpu().numpy().astype(np.float16)
                file_name = image_name.split('/')[-1]
                save_label = os.path.join(pre_mask_path, file_name)
                # cv2.imwrite(save_label, mask_im)
                # newpath= r'./result_picture/'+str(T)+"real1501.tif"
                # print(newpath)
                write_tif(save_label, mask_im, im_geotrans, im_proj, width, height, gdal.GDT_Int16)
                # print("写入{}成功".format(save_label))
                # save_visual = os.path.join(pre_vis_path, file_name)
                # print("开始写入{}".format(save_visual))
                # translabeltovisual(save_label, save_visual,num_classes)
                # print("写入{}成功".format(save_visual))

def translabeltovisual(save_label, path,num_classes):

    im = cv2.imread(save_label)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            pred_class=im[i][j][0]
            im[i][j] = num_classes[pred_class]
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, im)



if __name__ == "__main__":
    with open(r'predict_config1.json', encoding='utf-8') as f:
        config = json.load(f)
    num=int(config['num_classes'])
    # num_classes=[[255,215,0], [240,230,140], [218,165,32], [255,105,180], [0,255,127], [34,139,34], [143,188,143],[105,139,105],[24,72,45],[150,150,150],[60,129,139],[0,191,255]]

    predict(config)

    # newpath= r'./result_picture/'+str(T)+"real1501.tif"
    # print(newpath)
    # write_tif(newpath, predict_proba, im_geotrans, im_proj, width, height, gdal.GDT_Int16)

