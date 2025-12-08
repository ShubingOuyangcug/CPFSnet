import time
import os
import logging
from tqdm import tqdm
from sklearn.model_selection import KFold
from utils import mutiunet_dataset_eval
from models import unet
from models import hrnet
from models import Mymodel2
from models import Mymodeljustneibour
from models import unetFEGcn
from metrics import eval_metrics
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
from models import unet2ecoder
from models import unetbx
from models import unetbx2
from models import unetbrA
from models import unetbrB
from models import unetbrC
from models import unetbrD
from models import scnn
from models import scnn2
from models import PACSCNet
import copy
from models import logcanplus_model
# from predict import predict
# from lr_schedule import step_lr, exp_lr_scheduler

import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from models.lanenet.loss import DiscriminativeLoss, FocalLoss


def compute_loss(output, binary_seg_ret, binary_label, instance_label, device, loss_type='FocalLoss'):
    k_binary = 10  # 1.7
    k_instance = 0.3
    k_dist = 1.0

    if (loss_type == 'FocalLoss'):
        loss_fn = FocalLoss(device=device, gamma=2, alpha=[0.25, 0.75])
    elif (loss_type == 'CrossEntropyLoss'):
        loss_fn = nn.CrossEntropyLoss()
    else:
        # print("Wrong loss type, will use the default CrossEntropyLoss")
        loss_fn = nn.CrossEntropyLoss()

    binary_loss = loss_fn(output, binary_label)

    pix_embedding = binary_seg_ret
    ds_loss_fn = DiscriminativeLoss(0.5, 1.5, 1.0, 1.0, 0.001)
    var_loss, dist_loss, reg_loss = ds_loss_fn(pix_embedding, instance_label)
    binary_loss = binary_loss * k_binary
    var_loss = var_loss * k_instance
    dist_loss = dist_loss * k_dist
    instance_loss = var_loss + dist_loss
    total_loss = binary_loss + instance_loss
    # out = net_output["binary_seg_pred"]

    return total_loss


def dice_loss(logits, targets, smooth=1.0):
    outputs = F.softmax(logits, dim=1)  # 对预测值做softmax计算
    targets = torch.unsqueeze(targets, dim=1)  # 标签是3通道，增加一个通道，方便后续计算。
    targets = torch.zeros_like(logits).scatter_(dim=1, index=targets.type(torch.int64), src=torch.ones_like(
        logits))  # target标签中用1，2，3...分别代表第几类分割标签，现通过通道数表示标签类别

    inter = outputs * targets  # 计算两个标签的交集
    dice = 1 - ((2 * inter.sum(dim=(2, 3)) + smooth) / (outputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth))
    return dice.mean()


# def dice_loss(logits, targets, smooth=1.0):
#     # 对预测值进行sigmoid激活，使其在0到1之间
#     predicted = torch.sigmoid(logits)
#     targets = torch.unsqueeze(targets, dim=1)
#     # 计算交集
#     intersection = torch.sum(predicted * targets)
#
#     # 计算两倍的交集加上平滑值
#     dice_coefficient = (2.0 * intersection + smooth) / (torch.sum(predicted) + torch.sum(targets) + smooth)
#
#     # 返回Dice Loss，即1减去Dice系数
#     return 1.0 - dice_coefficient

class FocalLoss(nn.Module):
    '''
    Only consider two class now: foreground, background.
    '''

    def __init__(self, device, gamma=2, alpha=[0.5, 0.5], n_class=2, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.n_class = n_class
        self.device = device

    def forward(self, input, target):
        pt = F.softmax(input, dim=1)
        pt = pt.clamp(min=0.000001, max=0.999999)
        target_onehot = torch.zeros((target.size(0), self.n_class, target.size(1), target.size(2))).to(self.device)
        loss = 0
        for i in range(self.n_class):
            target_onehot[:, i, ...][target == i] = 1
        for i in range(self.n_class):
            loss -= self.alpha[i] * (1 - pt[:, i, ...]) ** self.gamma * target_onehot[:, i, ...] * torch.log(
                pt[:, i, ...])

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss


class CELoss(nn.Module):
    def __init__(self, loss_name=['CELoss', 'CELoss'], loss_weight=[1.0, 0.8], ignore_index=255, reduction='mean'):
        """
        根据LOGCAN++论文设计的损失函数

        Args:
            loss_name: 损失函数名称列表 ['CELoss', 'CELoss']
            loss_weight: 损失权重 [主损失权重, 辅助损失权重]
            ignore_index: 忽略的标签索引
            reduction: 损失减少方式
        """
        super(CELoss, self).__init__()

        self.loss_name = loss_name
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.reduction = reduction

        # 创建主损失函数
        if loss_name[0] == 'CELoss':
            self.main_criterion = nn.CrossEntropyLoss(
                ignore_index=self.ignore_index,
                reduction=self.reduction
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_name[0]}")

        # 创建辅助损失函数
        if loss_name[1] == 'CELoss':
            self.aux_criterion = nn.CrossEntropyLoss(
                ignore_index=self.ignore_index,
                reduction=self.reduction
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_name[1]}")

    def forward(self, pred, target):
        """
        前向传播计算损失

        Args:
            pred: 模型输出 [final_output, aux_output]
            target: 真实标签

        Returns:
            total_loss: 总损失
            loss_dict: 损失字典
        """
        if not isinstance(pred, (list, tuple)) or len(pred) != 2:
            raise ValueError("预测输出应该是包含两个元素的列表 [final_output, aux_output]")

        final_output, aux_output = pred

        # 计算主损失
        main_loss = self.main_criterion(final_output, target)

        # 计算辅助损失
        aux_loss = self.aux_criterion(aux_output, target)

        # 根据论文公式(10)计算总损失
        total_loss = self.loss_weight[0] * main_loss + self.loss_weight[1] * aux_loss

        # 返回损失字典
        loss_dict = {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'aux_loss': aux_loss
        }

        return total_loss

class adjacentLoss(nn.Module):
    '''
    Only consider two class now: foreground, background.
    '''

    def __init__(self, device, gamma=2, alpha=[0.5, 0.5], n_class=2, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.n_class = n_class
        self.device = device

    def forward(self, input, target):
        pt = F.softmax(input, dim=1)
        pt = pt.clamp(min=0.000001, max=0.999999)
        loss = 0
        for c in range(pt.size(2) - 2):
            diff_c = torch.abs((pt[:, 0, c, :] - pt[:, 0, c + 1, :]) - (pt[:, 0, c + 1, :] - pt[:, 0, c + 2, :]))
            loss += torch.sum(diff_c)
        for d in range(pt.size(3) - 2):
            diff_d = torch.abs((pt[:, 0, :, d] - pt[:, 0, :, d + 1]) - (pt[:, 0, :, d + 1] - pt[:, 0, :, d + 2]))
            loss += torch.sum(diff_d)

        return loss


def train(config, flag, labele, losse, modele):
    # train配置
    device = torch.device('cuda:0')
    selected2 = modele

    selected = selected2 + flag
    choose_loss = losse

    # 在训练开始前初始化logger，确保所有折使用同一个logger
    logger, log_file_path = initLogger(selected)
    fold_results = []

    # loss
    # criterion = dice_loss()
    # criterion = nn.CrossEntropyLoss()
    weight = torch.tensor([1, 6], device=device, dtype=torch.float)
    # criterion = nn.CrossEntropyLoss(weight)
    criterion = nn.CrossEntropyLoss(weight=weight)
    criterion2 = FocalLoss(device=device)
    criterion3 = adjacentLoss(device=device)
    criterion_CE = CELoss(
        loss_name=['CELoss', 'CELoss'],
        loss_weight=[1.0, 0.8],
        ignore_index=255
    )
    # train data

    dst_train = mutiunet_dataset_eval.UnetDataset(config['train_list'], labele, train=False, device=device)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 记录开始时间
    logger.info("开始5折交叉验证训练")
    logger.info(f"模型: {selected}, 损失函数: {choose_loss}")
    logger.info("=" * 80)

    for fold, (train_indices, val_indices) in enumerate(kf.split(dst_train)):
        print(selected)
        from models import PACSCNet
        if selected2 == 'logcanplus':
            model = logcanplus_model.LoGCANPlus(num_classes=config['num_classes'])
        if selected2 == 'PACSCNet':
            model = PACSCNet.FFNet()
        if selected2 == 'unetFEGcn':
            model = unetFEGcn.UNet(num_classes=config['num_classes'])
        if selected2 == 'unet':
            model = unet.UNet(num_classes=config['num_classes'])
        if selected2 == 'unetbx':
            model = unetbx.UNet(num_classes=config['num_classes'])
        if selected2 == 'unetbx2':
            model = unetbx2.UNet(num_classes=config['num_classes'])
        if selected2 == 'unetbrA':
            model = unetbrA.UNet(num_classes=config['num_classes'])
        if selected2 == 'unetbrB':
            model = unetbrB.UNet(num_classes=config['num_classes'])
        if selected2 == 'unetbrC':
            model = unetbrC.UNet(num_classes=config['num_classes'])
        if selected2 == 'unetbrD':
            model = unetbrD.UNet(num_classes=config['num_classes'])

        if selected2 == 'scnn':
            model = scnn.SCNN()
        if selected2 == 'scnn2':
            model = scnn2.SCNN()
        if selected2 == 'hrnet':
            model = hrnet.HighResolutionNet()
        if selected2 == 'unetPlusPlus':
            model = unetPlusPlus.UNetPlusPlus(num_classes=config['num_classes'])
        if selected2 == 'pspnet':
            model = pspnet.PSPNet()
        if selected2 == "attentionUnet":
            model = attentionUnet.AttentionUnet(num_classes=config['num_classes'])
        if selected2 == "deeplabv3":
            model = deeplabv3.DeepLab(2)
        if selected2 == "swinUnet":
            model = swinUnet.SwinTransformerSys()
        if selected2 == "bihrnet":
            model = bihrnet.HighResolutionNet(num_classes=config['num_classes'])
        if selected2 == 'mymodel':
            model = Mymodel2.MyModel(num_classes=config['num_classes'])
        if selected2 == 'mymodelcoordconv':
            model = Mymodel3.MyModel(num_classes=config['num_classes'])
        if selected2 == 'Mymodeljustneibour':
            model = Mymodeljur.MyModel(num_classes=config['num_classes'])
        if selected2 == 'Mymodelaverage':
            model = Mymodelaverage.MyModel(num_classes=config['num_classes'])
        if selected2 == "res_unet_plus":
            model = res_unet_plus.ResUnetPlusPlus(channel=7)
        if selected2 == "bisenetv2":
            model = bisenetv2.BiSeNetv2()
        if selected2 == "FTUNetFormer":
            model = FTUNetFormer.FTUNetFormer()
        if selected2 == "res_unet":
            model = res_unet.ResUnet()
        if selected2 == "unet2ecoder":
            model = unet2ecoder.UNet(num_classes=config['num_classes'])

        model.to(device)

        print(f"Fold {fold + 1}/5")
        logger.info(f"开始训练第 {fold + 1}/5 折")

        two_train = [dst_train[i] for i in train_indices]
        two_valid = [dst_train[i] for i in val_indices]
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        dataloader_train = DataLoader(two_train, shuffle=True, batch_size=config['batch_size'], drop_last=True)

        # validation data
        dataloader_valid = DataLoader(two_valid, batch_size=config['batch_size'])

        cur_acc = []
        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=[config['momentum'], 0.999],
                                     weight_decay=config['weight_decay'])
        # 最优val准确率，根据这个保存模型
        val_max_pixACC = 0.0
        val_max_mIoU = 0

        # 【新增】记录当前折次的最优结果
        best_fold_result = {
            'fold': fold,
            'epoch': 0,
            'mIoU': 0,
            'pixelAcc': 0,
            'loss': 0,
            'class_precision': None,
            'class_recall': None,
            'class_f1': None,
            'conf_matrix_val': None,
            'train_pixelAcc': 0,  # 新增训练精度记录
            'train_mIoU': 0  # 新增训练mIoU记录
        }

        for epoch in range(config['num_epoch']):
            epoch_start = time.time()
            # lr

            model.train()
            loss_sum = 0.0
            correct_sum = 0.0
            labeled_sum = 0.0
            inter_sum = 0.0
            unoin_sum = 0.0
            pixelAcc = 0.0
            IoU = 0.0
            tbar = tqdm(dataloader_train, ncols=120)

            # 混淆矩阵
            conf_matrix_train = np.zeros((config['num_classes'], config['num_classes']))

            for batch_idx, (data, background, target, target2, label_sence, path) in enumerate(tbar):  ###change
                tic = time.time()

                # data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data, background)  ###change

                if choose_loss == "cp":
                    loss = criterion(output, target2)
                elif choose_loss == "mcp":
                    loss = criterion(output, target2) + criterion(output2, target)
                elif choose_loss == "foc":
                    loss = criterion_CE(output, target2)
                elif choose_loss == "mm":
                    loss = criterion(output, target2) + criterion(output, target)
                elif choose_loss == "2mm":
                    loss = criterion(output1, target) + criterion(output2, target2)
                elif choose_loss == "cpaj":
                    loss = criterion(output, target2) + criterion3(output, target2) * 0.1
                elif choose_loss == "senc":

                    loss = criterion(output, target2) + criterion(h, label_sence)

                else:
                    binary_seg_ret = torch.argmax(F.softmax(output, dim=1), dim=1, keepdim=True)
                    loss = compute_loss(output, binary_seg_ret, target2, target2, device=device, loss_type='FocalLoss')

                loss_sum += loss.item()
                loss.backward()
                optimizer.step()
                if choose_loss == "foc":
                    correct, labeled, inter, unoin, conf_matrix_train = eval_metrics(output[0], target2, config['num_classes'],conf_matrix_train)
                else:
                    correct, labeled, inter, unoin, conf_matrix_train = eval_metrics(output, target2,
                                                                                     config['num_classes'],
                                                                                     conf_matrix_train)

                correct_sum += correct
                labeled_sum += labeled
                inter_sum += inter
                unoin_sum += unoin
                pixelAcc = 1.0 * correct_sum / (np.spacing(1) + labeled_sum)
                IoU = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)
                tbar.set_description(
                    'Fold {}/5 TRAIN ({}) | Loss: {:.5f} | OA {:.5f} mIoU {:.5f} | bt {:.2f} et {:.2f}|'.format(
                        fold + 1,
                        epoch, loss_sum / ((batch_idx + 1) * config['batch_size']),
                        pixelAcc, IoU.mean(),
                        time.time() - tic, time.time() - epoch_start))
                cur_acc.append(pixelAcc)

            logger.info('Fold {}/5 TRAIN ({}) | Loss: {:.5f} | OA {:.5f} IOU {}  mIoU {:.5f} '.format(fold + 1,
                                                                                                      epoch,
                                                                                                      loss_sum / ((
                                                                                                                              batch_idx + 1) *
                                                                                                                  config[
                                                                                                                      'batch_size']),
                                                                                                      pixelAcc,
                                                                                                      toString(IoU),
                                                                                                      IoU.mean()))

            # val
            test_start = time.time()

            model.eval()
            loss_sum = 0.0
            correct_sum = 0.0
            labeled_sum = 0.0
            inter_sum = 0.0
            unoin_sum = 0.0
            pixelAcc = 0.0
            mIoU = 0.0
            tbar = tqdm(dataloader_valid, ncols=120)
            class_precision = np.zeros(config['num_classes'])
            class_recall = np.zeros(config['num_classes'])
            class_f1 = np.zeros(config['num_classes'])
            # val_list=[]

            # data, target = data.to(device), target.to(device)
            with torch.no_grad():
                # 混淆矩阵
                conf_matrix_val = np.zeros((config['num_classes'], config['num_classes']))
                for batch_idx, (data, background, target, target2, label_sence, path) in enumerate(tbar):  ##change
                    tic = time.time()

                    output = model(data, background)

                    if choose_loss == "cp":
                        loss = criterion(output, target2)
                    elif choose_loss == "mcp":
                        loss = criterion(output, target2) + criterion(output2, target)
                    elif choose_loss == "foc":
                        loss = criterion_CE(output, target2)
                    elif choose_loss == "mm":
                        loss = criterion(output, target2) + criterion(output, target)
                    elif choose_loss == "2mm":
                        loss = criterion(output1, target) + criterion(output2, target2)
                    elif choose_loss == "cpaj":
                        loss = criterion(output, target2) + criterion3(output, target2) * 0.1
                    elif choose_loss == "senc":
                        loss = criterion(output, target2) + criterion(h, label_sence)
                    else:
                        binary_seg_ret = torch.argmax(F.softmax(output, dim=1), dim=1, keepdim=True)
                        loss = compute_loss(output, binary_seg_ret, target2, target2, device=device,
                                            loss_type='FocalLoss')

                    loss_sum += loss.item()

                    if choose_loss == "foc":
                        correct, labeled, inter, unoin, conf_matrix_val = eval_metrics(output[0], target2,
                                                                                       config['num_classes'],
                                                                                       conf_matrix_val)
                    else:
                        correct, labeled, inter, unoin, conf_matrix_val = eval_metrics(output, target2,
                                                                                       config['num_classes'],
                                                                                       conf_matrix_val)
                    correct_sum += correct
                    labeled_sum += labeled
                    inter_sum += inter
                    unoin_sum += unoin
                    pixelAcc = 1.0 * correct_sum / (np.spacing(1) + labeled_sum)
                    mIoU = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)

                    for i in range(config['num_classes']):
                        # 每一类的precision
                        class_precision[i] = 1.0 * conf_matrix_val[i, i] / conf_matrix_val[:, i].sum()
                        # 每一类的recall
                        class_recall[i] = 1.0 * conf_matrix_val[i, i] / conf_matrix_val[i].sum()
                        # 每一类的f1
                        class_f1[i] = (2.0 * class_precision[i] * class_recall[i]) / (
                                    class_precision[i] + class_recall[i])

                    tbar.set_description(
                        'Fold {}/5 VAL ({}) | vvLoss: {:.5f} | --------------------Acc {:.5f} vmIoU {:.5f} |vMf1{:.5f}| bt {:.2f} et {:.2f}|'.format(
                            fold + 1, epoch, loss_sum / ((batch_idx + 1) * config['batch_size']),
                            pixelAcc, mIoU.mean(), class_f1.mean(),
                            time.time() - tic, time.time() - test_start))

                # 【新增】记录当前epoch的验证结果，并更新最优结果
                current_val_loss = loss_sum / ((batch_idx + 1) * config['batch_size'])

                if mIoU.mean() > val_max_mIoU:
                    val_max_mIoU = mIoU.mean()
                    best_epoch = np.zeros(2)
                    best_epoch[0] = epoch
                    best_epoch[1] = mIoU.mean()

                    # 更新当前折的最优结果
                    best_fold_result.update({
                        'fold': fold,
                        'epoch': epoch,
                        'mIoU': mIoU.mean(),
                        'pixelAcc': pixelAcc,
                        'loss': current_val_loss,
                        'class_precision': copy.deepcopy(class_precision),
                        'class_recall': copy.deepcopy(class_recall),
                        'class_f1': copy.deepcopy(class_f1),
                        'conf_matrix_val': copy.deepcopy(conf_matrix_val),
                        'train_pixelAcc': pixelAcc,  # 记录训练精度
                        'train_mIoU': IoU.mean()  # 记录训练mIoU
                    })

                    if os.path.exists(config['save_model']['save_path']) is False:
                        os.mkdir(config['save_model']['save_path'])
                    torch.save(model.state_dict(),
                               os.path.join(config['save_model']['save_path'], selected + '_' + str(fold) + '_jx.pth'))
                    # np.savetxt(os.path.join(config['save_model']['save_path'], selected + '_conf_matrix_val.txt'),
                    #            conf_matrix_val, fmt="%d")
                    np.savetxt(
                        os.path.join(config['save_model']['save_path'], selected + '_' + str(fold) + '_best_epoch.txt'),
                        best_epoch)

            logger.info(
                'Fold {}/5 VAL ({}) | vvLoss: {:.5f} | --------------------ACC {:.5f} |IOU {} |vmIoU {:.5f} |class_precision {}| class_recall {} | class_f1 {}|mean_f1 {}'.format(
                    fold + 1, epoch, loss_sum / ((batch_idx + 1) * config['batch_size']),
                    pixelAcc, toString(mIoU), mIoU.mean(), toString(class_precision), toString(class_recall),
                    toString(class_f1), class_f1.mean()))

        # 【新增】在当前折训练结束后，记录该折的最优结果
        logger.info("=" * 80)
        logger.info(f"第 {fold + 1}/5 折训练完成，最优结果:")
        logger.info(f"最佳轮次: {best_fold_result['epoch']}")
        logger.info(f"验证集mIoU: {best_fold_result['mIoU']:.5f}")
        logger.info(f"验证集准确率: {best_fold_result['pixelAcc']:.5f}")
        logger.info(f"验证集损失: {best_fold_result['loss']:.5f}")
        logger.info(f"训练集准确率: {best_fold_result['train_pixelAcc']:.5f}")
        logger.info(f"训练集mIoU: {best_fold_result['train_mIoU']:.5f}")
        logger.info(f"各类精度: {toString(best_fold_result['class_precision'])}")
        logger.info(f"各类召回率: {toString(best_fold_result['class_recall'])}")
        logger.info(f"各类F1: {toString(best_fold_result['class_f1'])}")
        logger.info("=" * 80)

        # 保存当前折的最优结果
        fold_results.append(best_fold_result)
        del model

    # 【新增】找到全局最优结果（基于验证集mIoU）
    global_best_result = max(fold_results, key=lambda x: x['mIoU'])

    # 【新增】在日志文件末尾添加最高mIoU验证集精度的信息
    logger.info("=" * 80)
    logger.info("全局最优验证结果:")
    logger.info(f"最佳折次: {global_best_result['fold'] + 1}")
    logger.info(f"最佳轮次: {global_best_result['epoch']}")
    logger.info(f"最高mIoU: {global_best_result['mIoU']:.5f}")
    logger.info(f"对应准确率: {global_best_result['pixelAcc']:.5f}")
    logger.info(f"对应损失: {global_best_result['loss']:.5f}")
    logger.info(f"训练集准确率: {global_best_result['train_pixelAcc']:.5f}")
    logger.info(f"训练集mIoU: {global_best_result['train_mIoU']:.5f}")
    logger.info(f"各类精度: {toString(global_best_result['class_precision'])}")
    logger.info(f"各类召回率: {toString(global_best_result['class_recall'])}")
    logger.info(f"各类F1: {toString(global_best_result['class_f1'])}")
    logger.info("=" * 80)

    # 【新增】将最优折的结果输出到同名的txt文件中
    save_best_result_to_txt(global_best_result, log_file_path, selected, choose_loss)


def toString(IOU):
    """将IOU数组转换为字符串格式"""
    if IOU is None:
        return "{}"

    # 修复：确保IOU是可迭代对象
    try:
        result = '{'
        for i, num in enumerate(IOU):
            result += str(i) + ': ' + '{:.4f}, '.format(num)
        result += '}'
        return result
    except (TypeError, IndexError):
        # 如果IOU不是可迭代对象或者为空，返回空字典
        return "{}"


def save_best_result_to_txt(best_result, log_file_path, model_name, loss_function):
    """将最优结果保存到与log同名的txt文件中"""
    if log_file_path is None:
        return

    # 生成txt文件路径（与log文件同名，但扩展名为.txt）
    txt_file_path = log_file_path.replace('.log', '_best_result.txt')

    try:
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("最优折次训练结果汇总\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"模型名称: {model_name}\n")
            f.write(f"损失函数: {loss_function}\n")
            f.write(f"最优折次: {best_result['fold'] + 1}/5\n")
            f.write(f"最优轮次: {best_result['epoch']}\n\n")

            f.write("验证集性能:\n")
            f.write(f"  mIoU: {best_result['mIoU']:.6f}\n")
            f.write(f"  像素准确率: {best_result['pixelAcc']:.6f}\n")
            f.write(f"  损失: {best_result['loss']:.6f}\n\n")

            f.write("训练集性能:\n")
            f.write(f"  像素准确率: {best_result['train_pixelAcc']:.6f}\n")
            f.write(f"  mIoU: {best_result['train_mIoU']:.6f}\n\n")

            f.write("各类别详细指标:\n")
            f.write(f"  精度: {toString(best_result['class_precision'])}\n")
            f.write(f"  召回率: {toString(best_result['class_recall'])}\n")
            f.write(f"  F1分数: {toString(best_result['class_f1'])}\n\n")

            # 添加混淆矩阵信息（如果需要）
            if best_result['conf_matrix_val'] is not None:
                f.write("验证集混淆矩阵:\n")
                np.savetxt(f, best_result['conf_matrix_val'], fmt='%d')
                f.write("\n")

            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
            f.write("=" * 60 + "\n")

        print(f"最优结果已保存到: {txt_file_path}")
    except Exception as e:
        print(f"保存最优结果到txt文件时出错: {e}")


def initLogger(model_name):
    """初始化logger，返回logger实例和log文件路径"""
    logger = logging.getLogger()

    # 如果logger已经有handler，说明已经初始化过，直接返回
    if logger.handlers:
        # 尝试获取现有的log文件路径
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return logger, handler.baseFilename
        return logger, None

    logger.setLevel(logging.INFO)

    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = r'logs'
    # 确保日志目录存在
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_name = os.path.join(log_path, "new" + model_name + '_jx_new_metrics' + rq + '.log')
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger, log_name


if __name__ == '__main__':
    # train(train_config.json)
    while True:
        print(1)