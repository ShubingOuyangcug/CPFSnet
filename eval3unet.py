import os
import cv2
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models import unetFEGcn
from utils.mutiunet_dataset_eval import read_tiff, read_image
from osgeo import gdal
from models import unet
from metrics import eval_metrics
from train1kfold import toString
import os
from metrics import eval_metrics
import numpy as np
import torch
from torchvision import transforms
from models import unetbx
from models import unetbx2
from models import Mymodeljustneibour
from models import Mymodelaverage
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
from models import unetbrA
from models import unetbrB
from models import unetbrC
from models import unetbrD
from models import scnn
from models import scnn2
from models import PACSCNet
from models import logcanplus_model

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


def eval(config, flag, labele, modele):
    # 创建test文件夹
    os.makedirs('test', exist_ok=True)

    # 创建输出文件
    output_file = os.path.join('test', f'{modele}_{flag}.txt')

    device = torch.device('cuda:0')
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    selected = modele
    if selected == 'logcanplus':
        model = logcanplus_model.LoGCANPlus(num_classes=config['num_classes'])
    if selected == 'PACSCNet':
        model = PACSCNet.FFNet()
    if selected == 'unetFEGcn':
        model = unetFEGcn.UNet(num_classes=config['num_classes'])
    if selected == 'unet':
        model = unet.UNet(num_classes=config['num_classes'])
    if selected == 'mymodelcoordconv':
        model = Mymodel3.MyModel(num_classes=config['num_classes'])
    if selected == 'unetbx':
        model = unetbx.UNet(num_classes=config['num_classes'])
    if selected == 'scnn2':
        model = scnn2.SCNN()
    if selected == 'unetbx2':
        model = unetbx2.UNet(num_classes=config['num_classes'])
    if selected == 'scnn':
        model = scnn.SCNN()
    if selected == 'unetbrA':
        model = unetbrA.UNet(num_classes=config['num_classes'])
    if selected == 'unetbrB':
        model = unetbrB.UNet(num_classes=config['num_classes'])
    if selected == 'unetbrC':
        model = unetbrC.UNet(num_classes=config['num_classes'])
    if selected == 'unetbrD':
        model = unetbrD.UNet(num_classes=config['num_classes'])
    if selected == 'unetPlusPlus':
        model = unetPlusPlus.UNetPlusPlus(num_classes=config['num_classes'])
    if selected == 'pspnet':
        model = pspnet.PSPNet()
    if selected == "attentionUnet":
        model = attentionUnet.AttentionUnet(num_classes=config['num_classes'])
    if selected == "deeplabv3":
        model = deeplabv3.DeepLab(2)
    if selected == 'Mymodeljustneibour':
        model = Mymodeljustneibour.MyModel(num_classes=config['num_classes'])
    if selected == "swinUnet":
        model = swinUnet.SwinTransformerSys()
    if selected == 'Mymodelaverage':
        model = Mymodelaverage.MyModel(num_classes=config['num_classes'])
    if selected == "bihrnet":
        model = bihrnet.HighResolutionNet(num_classes=config['num_classes'])
    if selected == "res_unet_plus":
        model = res_unet_plus.ResUnetPlusPlus(channel=11)
    if selected == "bisenetv2":
        model = bisenetv2.BiSeNetv2()
    if selected == "FTUNetFormer":
        model = FTUNetFormer.FTUNetFormer()
    if selected == "res_unet":
        model = res_unet.ResUnet()
    selected = selected + flag

    # 打开文件准备写入
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(5):
            check_point = os.path.join(config['save_model']['save_path'], selected + '_' + str(i) + '_jx.pth')

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(torch.load(check_point, map_location=device), False)
            model.cuda()
            model.eval()
            # 混淆矩阵
            conf_matrix_test = np.zeros((config['num_classes'], config['num_classes']))

            correct_sum = 0.0
            labeled_sum = 0.0
            inter_sum = 0.0
            unoin_sum = 0.0
            pixelAcc = 0.0
            mIoU = 0.0

            class_precision = np.zeros(config['num_classes'])
            class_recall = np.zeros(config['num_classes'])
            class_f1 = np.zeros(config['num_classes'])
            with open(config['img_txt'], 'r', encoding='utf-8') as f_txt:
                for line in f_txt.readlines():

                    image_name1, label_name = line.strip().split()
                    image_name = image_name1.replace("image", "bigimage")  # coldem11bandgb

                    label_name = label_name.replace("label", labele)  # 5wstructure400mDU

                    root_dir = ''
                    image_name = os.path.join(root_dir, image_name)

                    background_path = image_name1.replace("image", "bigdatabgimage")  # backgroundgb

                    label_name = os.path.join(root_dir, label_name)
                    label = torch.from_numpy(np.asarray(read_label(label_name), dtype=np.int32)).long().cuda()

                    randN1 = 0
                    image = read_image(image_name, randN1, device, augmentation=False)
                    # 加一维,batch_size=1
                    image = image.unsqueeze(0)

                    image_background = read_image(background_path, randN1, device, augmentation=False)
                    image_background = image_background.unsqueeze(0)

                    # labelmulti

                    output = model(image, image_background)
                    if selected == 'logcanplus':
                        output = output[0]
                    elif selected == 'unetbx2':
                        batch_size, num_channels, height, width = output.shape
                        output = output[batch_size // 2:]
                    # _, pred = output.max(1)
                    # pred = pred.view(256, 256)
                    # mask_im = pred.cpu().numpy().astype(np.uint8)
                    correct, labeled, inter, unoin, conf_matrix_test = eval_metrics(output, label,
                                                                                    config['num_classes'],
                                                                                    conf_matrix_test)
                    correct_sum += correct
                    labeled_sum += labeled
                    inter_sum += inter
                    unoin_sum += unoin
                    pixelAcc = 1.0 * correct_sum / (np.spacing(1) + labeled_sum)
                    mIoU = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)

                    for j in range(config['num_classes']):
                        # 每一类的precision
                        class_precision[j] = 1.0 * conf_matrix_test[j, j] / conf_matrix_test[:, j].sum()
                        # 每一类的recall
                        class_recall[j] = 1.0 * conf_matrix_test[j, j] / conf_matrix_test[j].sum()
                        # 每一类的f1
                        class_f1[j] = (2.0 * class_precision[j] * class_recall[j]) / (
                                    class_precision[j] + class_recall[j])

            # 将输出同时写入文件和打印到控制台
            output_line = 'OA {:.5f} |IOU {} |mIoU {:.5f} |class_precision {}| class_recall {} | class_f1 {}|'.format(
                pixelAcc, toString(mIoU), mIoU.mean(), toString(class_precision), toString(class_recall),
                toString(class_f1))
            print(output_line)
            f.write(output_line + '\n')

            tttxxx = 'OA {:.5f} |IOU {} |mIoU {:.5f} |class_precision {}| class_recall {} | class_f1 {}|'.format(
                pixelAcc, toString(mIoU), mIoU.mean(), toString(class_precision), toString(class_recall),
                toString(class_f1))
            np.savetxt(os.path.join("confuse_matrix", selected + '_jx_matrix_test.txt'), conf_matrix_test, fmt="%d")

            with open(os.path.join("confuse_matrix", selected + '_test.txt'), 'a') as file:
                file.write("\n")
                file.write(tttxxx)


if __name__ == "__main__":
    with open(r'eval_confignuet.json', encoding='utf-8') as f:
        config = json.load(f)
    flag = "400m11band"
    labele = "your_label_here"  # 需要提供labele参数
    modele = "your_model_here"  # 需要提供modele参数
    eval(config, flag, labele, modele)