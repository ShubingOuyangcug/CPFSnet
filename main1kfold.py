# coding: utf-8

import json
from train1kfold import train
from eval3unet import eval
from predict1 import predict


#unetbx是提出的模型
label_list = ["bd1203"]
loss_list = ["cp", "foc", "muti",'mm','2mm',"cpaj","senc","mcp"]
model_list = ["PACSCNet","scnn2","scnn","unet","unetbx", "FTUNetFormer","unetbrA","unetbrB","unetbrC","unetPlusPlus", "pspnet", "attentionUnet", "deeplabv3", "swinUnet", "bihrnet", "res_unet_plus",
          "bisenetv2", "res_unet", "unet2ecoder","unetbrD","logcanplus"] #8unetPlusPlus， 9 pspnet 10batch会out cuda  #11deeplabv3会报错，要在后面加一个,drop_last=True dataloader_train = DataLoader(two_train, shuffle=True, batch_size=config['batch_size'],drop_last=True)
#12swinUnet vMf1nanB NOASPP



for i in [7]:#11,9没跑 #4FTUNetFormer，8unetPlusPlus,10attentionUnet,9pspnet,11,12,14
    label = label_list[0]
    loss = loss_list[0]
    model = model_list[i]
    flag = label + loss + "_1"
    print(model + "_" + flag)
    with open(r'train_configkfold.json', encoding="utf-8") as f:
        config = json.load(f)
    train(config, flag, label, loss, model)

    print(flag)
    with open(r'eval_confignuet.json', encoding='utf-8') as f:
        config = json.load(f)
    eval(config, flag, label, model)
    print(model + "_" + flag)

    with open(r'predict_config1.json', encoding='utf-8') as f:
        config = json.load(f)
    predict(config, flag, model)

    print(flag, "-----------------------------------------------------------------------------------")

# for i in [9]:#11,9没跑 #4FTUNetFormer，8unetPlusPlus,10attentionUnet,9pspnet,11,12,14
#     label = label_list[0]
#     loss = loss_list[0]
#     model = model_list[i]
#     flag = label + loss + "_2_pbx"
#     print(model + "_" + flag)
#     with open(r'train_configkfold.json', encoding="utf-8") as f:
#         config = json.load(f)
#     train(config, flag, label, loss, model)
#
#     print(flag)
#     with open(r'eval_confignuet.json', encoding='utf-8') as f:
#         config = json.load(f)
#     eval(config, flag, label, model)
#     print(model + "_" + flag)
#
#     with open(r'predict_config1.json', encoding='utf-8') as f:
#         config = json.load(f)
#     predict(config, flag, model)
#
#     print(flag, "-----------------------------------------------------------------------------------")
#
# for i in [9]:#11,9没跑 #4FTUNetFormer，8unetPlusPlus,10attentionUnet,9pspnet,11,12,14
#     label = label_list[0]
#     loss = loss_list[0]
#     model = model_list[i]
#     flag = label + loss + "_3_pbx"
#     print(model + "_" + flag)
#     with open(r'train_configkfold.json', encoding="utf-8") as f:
#         config = json.load(f)
#     train(config, flag, label, loss, model)
#
#     print(flag)
#     with open(r'eval_confignuet.json', encoding='utf-8') as f:
#         config = json.load(f)
#     eval(config, flag, label, model)
#     print(model + "_" + flag)
#
#     with open(r'predict_config1.json', encoding='utf-8') as f:
#         config = json.load(f)
#     predict(config, flag, model)
#
#     print(flag, "-----------------------------------------------------------------------------------")
#
# for i in [9]:#11,9没跑 #4FTUNetFormer，8unetPlusPlus,10attentionUnet,9pspnet,11,12,14
#     label = label_list[0]
#     loss = loss_list[0]
#     model = model_list[i]
#     flag = label + loss + "_4_pbx"
#     print(model + "_" + flag)
#     with open(r'train_configkfold.json', encoding="utf-8") as f:
#         config = json.load(f)
#     train(config, flag, label, loss, model)
#
#     print(flag)
#     with open(r'eval_confignuet.json', encoding='utf-8') as f:
#         config = json.load(f)
#     eval(config, flag, label, model)
#     print(model + "_" + flag)
#
#     with open(r'predict_config1.json', encoding='utf-8') as f:
#         config = json.load(f)
#     predict(config, flag, model)
#
#     print(flag, "-----------------------------------------------------------------------------------")
#
# for i in [9]:#11,9没跑 #4FTUNetFormer，8unetPlusPlus,10attentionUnet,9pspnet,11,12,14
#     label = label_list[0]
#     loss = loss_list[0]
#     model = model_list[i]
#     flag = label + loss + "_5_pbx"
#     print(model + "_" + flag)
#     with open(r'train_configkfold.json', encoding="utf-8") as f:
#         config = json.load(f)
#     train(config, flag, label, loss, model)
#
#     print(flag)
#     with open(r'eval_confignuet.json', encoding='utf-8') as f:
#         config = json.load(f)
#     eval(config, flag, label, model)
#     print(model + "_" + flag)
#
#     with open(r'predict_config1.json', encoding='utf-8') as f:
#         config = json.load(f)
#     predict(config, flag, model)
#
#     print(flag, "-----------------------------------------------------------------------------------")






