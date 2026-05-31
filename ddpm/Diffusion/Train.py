
import os
from datetime import datetime
from typing import Dict

import torch
import torch.optim as optim
import torchvision
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
import pickle
import numpy as np
from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer

from Scheduler import GradualWarmupScheduler


criterion_MSE = nn.MSELoss()
bias = 300  # bias: loss2/loss1
class_num = 100
latent_variable_dim = 512
CIL_step = 10

save_dir = os.path.join('./Checkpoints/unPEDCC_POD/CIFAR100', f'{class_num}_bias_1-{bias}_FC(Augm(0.5)_X0_t=Tstep_Cemb_TwightPOD_labels')
load_classify_dir = os.path.join('./Checkpoints/unPEDCC_POD/CIL_CIFAR100', f'{class_num-CIL_step}_bias_1-{bias}_FC(Augm(0.5)_X0_t=Tstep_Cemb_TwightPOD_labels')
load_generator_dir = save_dir

PEDCC_ui=os.path.join('./PEDCC/center_pedcc/', f'{class_num}_{latent_variable_dim}_s.pkl')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def read_pkl():
    f = open(PEDCC_ui, 'rb')
    print(f'PEDCC_MAP: {PEDCC_ui}')
    a = pickle.load(f)
    f.close()
    return a


# consieLinear层 实现了norm的fea与norm weight的点积计算，服务于margin based softmax loss
# 将w替换成pedcc，固定
# 计算余弦距离
class CosineLinear_PEDCC(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineLinear_PEDCC, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features), requires_grad=False)
        map_dict = read_pkl()
        tensor_empty = torch.Tensor([]).cuda()
        for label_index in range(self.out_features):
            tensor_empty = torch.cat((tensor_empty, map_dict[label_index].float().cuda()), 0)
        label_40D_tensor = tensor_empty.view(-1, self.in_features).permute(1, 0)
        label_40D_tensor = label_40D_tensor.cuda()
        self.weight.data = label_40D_tensor
        #print(self.weight.data)

    def forward(self, input):
        x = input  # size=(B,F)    F is feature len
        w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features
        cos_theta = x.mm(w)  # size=(B,Classnum)  x.dot(ww)

        return cos_theta  # size=(B,Classnum)]

#####NaCLoss##################
def NaCLoss(input, target, delta, t_index):
    ret_before = input * target
    ret_before = torch.sum(ret_before, dim=1).view(-1, 1)

    add_feature = delta * torch.ones((input.shape[0], 1)).cuda()
    input_after = torch.cat((input, add_feature), dim=1)
    input_after_norm = torch.norm(input_after, p=2, dim=1, keepdim=True)

    ret = ret_before / input_after_norm
    # threshold t=500, when t>500, t =1000
    # only calc loss2 for classify, not influence loss1 for genaration
    # t_index = torch.where(t_index > 500, torch.tensor(999).to(t_index.device), t_index)
    ret_t = ret.view(-1) * (1000 - t_index) / 1000
    ret_t = 1 - ret_t
    ret_t = ret_t.pow(2)
    ret_t = torch.mean(ret_t)

    return ret_t


#####SCLoss#########################
def SCLoss(map_PEDCC, label, feature):
    average_feature = map_PEDCC[label.long().data].float().cuda()
    feature_norm = l2_norm(feature)
    feature_norm = feature_norm - average_feature
    covariance100 = 1 / (feature_norm.shape[0] - 1) * torch.mm(feature_norm.T, feature_norm).float()
    covariance100_loss = torch.sum(pow(covariance100, 2)) - torch.sum(pow(torch.diagonal(covariance100), 2))
    covariance100_loss = covariance100_loss / (covariance100.shape[0] - 1)
    return covariance100_loss


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def get_cc_ic(output, label, ui, train=True):
    total = output.shape[0]
    total_distance = list()
    for i in range(total):
        distance_list = list()
        for ui_label in ui.values():
            distance = torch.sum((ui_label.float().cuda()-output[i])**2)
            distance_list.append(distance.item())
        idx = distance_list.index(min(distance_list))
        total_distance.append(idx)
    pred_label = torch.Tensor(total_distance).long().cuda()
    # if train is False:
    #     print(f'labels = {label}')
    #     print(f'pred_label = {pred_label}')
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def get_cc_ic_PEDCC(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    # print(f'pred_label= {pred_label}')
    # print(f'label= {label}')
    return num_correct / total


def train(modelConfig: Dict):
    from Diffusion.ModelCondition import UNet
    device = torch.device(modelConfig["device"])
    # sub_dir = modelConfig["lr"]

    # dataset
    CIFAR10_Train_ROOT = r'F:\dataset\CIFAR10\CIFAR10\train'
    CIFAR10_ddim250_e10w_Train_ROOT = r'F:\dataset\CIFAR10\ddim250_e=10w'
    CIFAR10_ddim250_e15w_Train_ROOT = r'F:\dataset\CIFAR10\ddim250_e=15w'
    CIFAR10_Test_ROOT = r'F:\dataset\CIFAR10\CIFAR10\test'
    # CIFAR100_Train_ROOT = '../data/diffgen_class_imgs/'
    # CIFAR100_Train_ROOT = r'F:\dataset\CIFAR100\CIFAR100_stage1/train/'
    CIFAR100_Train_ROOT = r'F:\dataset\CIFAR100\CIFAR100\train'
    CIFAR100_Test_ROOT = r'F:\dataset\CIFAR100\CIFAR100\test'
    CIFAR100_50_Train_ROOT =  r'F:\dataset\CIFAR100\CIFAR100_50\train'
    CIFAR100_50_Test_ROOT =  r'F:\dataset\CIFAR100\CIFAR100_50\test'

    # CIFAR100_50_Train_ROOT = r'F:\dataset\CIFAR100\CIFAR100_50\train'
    # CIFAR100_50_Test_ROOT = r'F:\dataset\CIFAR100\CIFAR100_50\test'
    # # CIFAR10_Test_ROOT = '/home/data/WSJ/dataset/CIFAR10/test'
    # # CIFAR10_Train_ROOT = '/home/data/WSJ/dataset/CIFAR10/train'
    # # CIFAR10_Train_ROOT = '/home/data/WSJ/Code/ddpm(copy)/SampledImgs_diff/diffgen_true'
    # CIFAR100_Train_ROOT = '/home/data/WSJ/dataset/CIFAR100/train'
    # CIFAR100_Test_ROOT = '/home/data/WSJ/dataset/CIFAR100/test'
    # CIFAR100_50_Train_ROOT = '/home/data/WSJ/dataset/CIFAR100/CIFAR100_50/train'
    # CIFAR100_50_Test_ROOT = '/home/data/WSJ/dataset/CIFAR100/CIFAR100_50/test'
    # CIFAR100_60_Train_ROOT = '/home/data/WSJ/dataset/CIFAR100/CIL_CIFAR100_60/train'
    # CIFAR100_60_Test_ROOT = '/home/data/WSJ/dataset/CIFAR100/CIL_CIFAR100_60/test'
    # CIFAR10_train_data = torchvision.datasets.ImageFolder(CIFAR10_Train_ROOT,
    #                                                        transform=transforms.Compose([
    #                                                            transforms.RandomHorizontalFlip(),
    #                                                            transforms.RandomCrop(size=32, padding=4),
    #                                                            transforms.ToTensor(),
    #                                                            transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                                                                                 std=[0.5, 0.5, 0.5]),
    #                                                        ])
    #                                                        )
    # CIFAR10_test_data = torchvision.datasets.ImageFolder(CIFAR10_Test_ROOT,
    #                                                       transform=transforms.Compose([
    #                                                           transforms.ToTensor(),
    #                                                           transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                                                                                std=[0.5, 0.5, 0.5]),
    #                                                       ])
    #                                                       )
    CIFAR100_train_data = torchvision.datasets.ImageFolder(CIFAR100_Train_ROOT,
                                                           transform=transforms.Compose([
                                                               transforms.RandomHorizontalFlip(),
                                                               transforms.RandomCrop(size=32, padding=4),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                    std=[0.5, 0.5, 0.5]),
                                                           ])
                                                           )
    CIFAR100_test_data = torchvision.datasets.ImageFolder(CIFAR100_Test_ROOT,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                   std=[0.5, 0.5, 0.5]),
                                                          ])
                                                          )
    CIFAR100_50_train_data = torchvision.datasets.ImageFolder(CIFAR100_50_Train_ROOT,
                                                           transform=transforms.Compose([
                                                               transforms.RandomHorizontalFlip(),
                                                               transforms.RandomCrop(size=32, padding=4),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                    std=[0.5, 0.5, 0.5]),
                                                           ])
                                                           )
    CIFAR100_50_test_data = torchvision.datasets.ImageFolder(CIFAR100_50_Test_ROOT,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                   std=[0.5, 0.5, 0.5]),
                                                          ])
                                                          )

    train_data = CIFAR100_train_data
    test_data = CIFAR100_test_data
    train_dataloader = DataLoader(
        train_data, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    test_dataloader = DataLoader(
        test_data, batch_size=modelConfig["test_batch_size"], shuffle=False, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = UNet(T=modelConfig["T"], num_labels=modelConfig["num_labels"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"])#.to(device)
    print(net_model)
    # load pre-trained Unet
    if modelConfig["training_load_classify_weight"]:
        load_weight = torch.load(os.path.join(load_generator_dir, 'classify_ckpt_300_.pt'), map_location='cpu')
        net_model.load_state_dict({k.replace('module.', ''): v for k, v in load_weight.items()})
        print(f'loading classify ckpt......')
    params = []
    for name, param in net_model.named_parameters():
        if 'head' in name:
            params.append(param)
        if 'time_embedding' in name:
            params.append(param)
        if 'cond_embedding' in name:
            params.append(param)
        if 'downblocks.12' in name:
            params.append(param)
        if 'downblocks.13' in name:
            params.append(param)
        if 'downblocks.14' in name:
            params.append(param)
        if 'middleblocks' in name:
            params.append(param)
        if 'upblocks' in name:
            params.append(param)
        if 'tail' in name:
            params.append(param)
    optimizer = torch.optim.AdamW(
        params, lr=modelConfig["lr"], weight_decay=1e-4)
    # optimizer = torch.optim.AdamW(
    #         net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    map_dict = read_pkl()
    fc = CosineLinear_PEDCC(latent_variable_dim, class_num)

    # start training
    map_PEDCC = torch.Tensor([])
    for i in range(class_num):
        map_PEDCC = torch.cat((map_PEDCC, map_dict[i].float()), 0)
    map_PEDCC = map_PEDCC.view(class_num, -1)  # (class_num, dimension)
    delta = 0.05
    for e in range(modelConfig["epoch"]):
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_acc = 0

        prev_time = datetime.now()
        l2_norm_trainSample = torch.Tensor([]).cuda()
        with tqdm(train_dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                    optimizer.zero_grad()
                    x_0 = images.cuda()
                    labels = labels.cuda()
                    # diff_labels : for Noneps and eps
                    diff_labels = labels + 1
                    tensor_empty1 = map_PEDCC[labels].float().cuda()
                    label_mse_tensor = tensor_empty1.view(labels.shape[0], -1)  # (b, 256)
                    label_mse_tensor = label_mse_tensor.cuda()
                    if np.random.rand() < 0.1:
                        diff_labels = torch.zeros_like(diff_labels).to(labels.device)

                    # print(diff_labels)
                    noise_mse, feature_vector, t_index = trainer(x_0, diff_labels)  # feature [b 256]

                    output = fc(feature_vector)  # out [b 10]
                    assert output.shape == (x_0.size(0), class_num)

                    with torch.no_grad():
                        l2_norm_trainSample = torch.cat((l2_norm_trainSample, torch.norm(output, p=2, dim=1)),
                                                        dim=0)  # [b]
                    loss2 = NaCLoss(feature_vector, label_mse_tensor, delta, t_index) * 200
                    loss3 = SCLoss(map_PEDCC, labels, feature_vector) * 200  # [class_num 256]  [b]  [b 256]
                    loss1 = noise_mse.sum() / 1000

                    #  print(f'loss1_noise: {loss1}')
                    # assert feature_vector.shape == label_tensor.shape           # [b 1024]
                    # loss2 = F.mse_loss(feature_norm, label_tensor, reduction='none').sum() / 1000.
                    # print(f'loss2_featurre: {loss2}')
                    loss = loss2 + loss3 + loss1
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        net_model.parameters(), modelConfig["grad_clip"])
                    optimizer.step()
                    train_loss += loss.item()
                    train_loss1 += loss1.item()
                    train_loss2 += loss2.item()
                    train_loss3 += loss3.item()
                    train_acc += get_cc_ic_PEDCC(output, labels)  # is output
        warmUpScheduler.step()
        if e % 200 == 0 and e > 0:
            torch.save(net_model.state_dict(), os.path.join(save_dir, f'ckpt_{e}_.pt'))
        curr_time = datetime.now()
        h, remainder = divmod((curr_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = " Time %02d:%02d:%02d" % (h, m, s)

        # test_dataloader = None
        if test_dataloader is not None:
            valid_loss = 0
            valid_acc = 0
            val_loss1 = 0
            val_loss2 = 0
            val_loss3 = 0
            net_model = net_model.eval()
            trainer = trainer.eval()
            for image, labels in tqdm(test_dataloader, desc="Processing val data: "):
                if torch.cuda.is_available():
                    x_0 = image.cuda()
                    labels = labels.cuda()
                    diff_labels = labels + 1
                    tensor_empty1 = map_PEDCC[labels].float().cuda()
                    label_mse_tensor = tensor_empty1.view(labels.shape[0], -1)  # (b, 256)
                    label_mse_tensor = label_mse_tensor.cuda()

                if np.random.rand() < 0.1:
                    diff_labels = torch.zeros_like(diff_labels).to(labels.device)

                # print(diff_labels)
                noise_mse, feature_vector, t_index = trainer(x_0, diff_labels, False)  # feature [b 256]

                assert feature_vector.shape == (x_0.size(0), latent_variable_dim)
                output = fc(feature_vector)  # out [b 10]
                assert output.shape == (x_0.size(0), class_num)

                with torch.no_grad():
                    l2_norm_trainSample = torch.cat((l2_norm_trainSample, torch.norm(output, p=2, dim=1)),
                                                    dim=0)  # [b]
                    loss2 = NaCLoss(feature_vector, label_mse_tensor, delta, t_index) * 200
                    loss3 = SCLoss(map_PEDCC, labels, feature_vector) * 200  # [class_num 256]  [b]  [b 256]

                    loss1 = noise_mse.sum() / 1000
                # loss2 = F.mse_loss(feature_vector, label_tensor, reduction='none').sum(dim=1)
                # assert loss2.shape == t_index.shape
                # loss2 = loss2 * (1 / (1 + t_index))
                # loss2 = loss2.sum() / 1000. * bias * 100
                # print(f'loss2.shape:{loss2}')
                # print(f'loss2_featurre: {loss2}')
                loss = loss1 + loss2 + loss3

                valid_loss += loss.item()
                val_loss1 += loss1.item()
                val_loss2 += loss2.item()
                val_loss3 += loss3.item()
                # mistake: feature_vector
                valid_acc += get_cc_ic_PEDCC(output, labels)

            epoch_str = ("Epoch %d TrainLoss: %f TrainAcc: %f ValidLoss: %f ValidAcc: %f"
                     % (e, train_loss / len(train_dataloader), train_acc / len(train_dataloader),
                        valid_loss / len(test_dataloader), valid_acc / len(test_dataloader)))
            Loss = ("TrainLoss1: %f TrainLoss2: %f Val_Loss1: %f Val_Loss2: %f"
                    % (train_loss1 / len(train_dataloader), train_loss2 / len(train_dataloader),
                       val_loss1 / len(test_dataloader), val_loss2 / len(test_dataloader)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train.Acc: %f,  "
                         % (e, train_loss / len(train_dataloader), train_acc / len(train_dataloader)))
            Loss = ("Train Loss1: %f, Train Loss2: %f,"
                    % (train_loss1 / len(train_dataloader), train_loss2 / len(train_dataloader)))
        prev_time = curr_time
        f = open(os.path.join(save_dir, f'log_diff_PEDCC_1-{bias}.txt'), 'a+')
        if e == 0:
            f.write(f'begin_time: {curr_time}'.center(100, '=') + '\n')
        print(" ")
        print(epoch_str + time_str)
        print(Loss + ", " + f'LR: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        f.write(epoch_str + time_str + '\n')
        f.write(Loss + ' ' + f'LR: {optimizer.state_dict()["param_groups"][0]["lr"]}' + '\n')
        if e == modelConfig["epoch"] - 1:
            f.write(f'end_time: {curr_time}'.center(100, '=') + '\n')
        f.close()


def classify(modelConfig: Dict):
    from Diffusion.ModelCondition import UNet
    device = torch.device(modelConfig["device"])

    # dataset
    CIFAR10_Train_ROOT = r'F:\dataset\CIFAR10\CIFAR10\train'
    CIFAR10_ddim250_e10w_Train_ROOT = r'F:\dataset\CIFAR10\ddim250_e=10w'
    CIFAR10_ddim250_e15w_Train_ROOT = r'F:\dataset\CIFAR10\ddim250_e=15w'
    CIFAR10_Test_ROOT = r'F:\dataset\CIFAR10\CIFAR10\test'
    # CIFAR100_Train_ROOT = '../data/diffgen_class_imgs/'
    # CIFAR100_Train_ROOT = r'F:\dataset\CIFAR100\CIFAR100_stage1/train/'
    CIFAR100_Train_ROOT = r'F:\dataset\CIFAR100\CIFAR100\train'
    CIFAR100_Test_ROOT = r'F:\dataset\CIFAR100\CIFAR100\test'
    CIFAR100_50_Train_ROOT =  r'F:\dataset\CIFAR100\CIFAR100_50\train'
    CIFAR100_50_Test_ROOT =  r'F:\dataset\CIFAR100\CIFAR100_50\test'

    # CIFAR100_50_Train_ROOT = r'F:\dataset\CIFAR100\CIFAR100_50\train'
    # CIFAR100_50_Test_ROOT = r'F:\dataset\CIFAR100\CIFAR100_50\test'
    # # CIFAR10_Test_ROOT = '/home/data/WSJ/dataset/CIFAR10/test'
    # # CIFAR10_Train_ROOT = '/home/data/WSJ/dataset/CIFAR10/train'
    # # CIFAR10_Train_ROOT = '/home/data/WSJ/Code/ddpm(copy)/SampledImgs_diff/diffgen_true'
    # CIFAR100_Train_ROOT = '/home/data/WSJ/dataset/CIFAR100/train'
    # CIFAR100_Test_ROOT = '/home/data/WSJ/dataset/CIFAR100/test'
    # CIFAR100_50_Train_ROOT = '/home/data/WSJ/dataset/CIFAR100/CIFAR100_50/train'
    # CIFAR100_50_Test_ROOT = '/home/data/WSJ/dataset/CIFAR100/CIFAR100_50/test'
    # CIFAR100_60_Train_ROOT = '/home/data/WSJ/dataset/CIFAR100/CIL_CIFAR100_60/train'
    # CIFAR100_60_Test_ROOT = '/home/data/WSJ/dataset/CIFAR100/CIL_CIFAR100_60/test'
    # CIFAR10_train_data = torchvision.datasets.ImageFolder(CIFAR10_Train_ROOT,
    #                                                        transform=transforms.Compose([
    #                                                            transforms.RandomHorizontalFlip(),
    #                                                            transforms.RandomCrop(size=32, padding=4),
    #                                                            transforms.ToTensor(),
    #                                                            transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                                                                                 std=[0.5, 0.5, 0.5]),
    #                                                        ])
    #                                                        )
    # CIFAR10_test_data = torchvision.datasets.ImageFolder(CIFAR10_Test_ROOT,
    #                                                       transform=transforms.Compose([
    #                                                           transforms.ToTensor(),
    #                                                           transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                                                                                std=[0.5, 0.5, 0.5]),
    #                                                       ])
    #                                                       )
    CIFAR100_train_data = torchvision.datasets.ImageFolder(CIFAR100_Train_ROOT,
                                                           transform=transforms.Compose([
                                                               transforms.RandomHorizontalFlip(),
                                                               transforms.RandomCrop(size=32, padding=4),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                    std=[0.5, 0.5, 0.5]),
                                                           ])
                                                           )
    CIFAR100_test_data = torchvision.datasets.ImageFolder(CIFAR100_Test_ROOT,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                   std=[0.5, 0.5, 0.5]),
                                                          ])
                                                          )
    CIFAR100_50_train_data = torchvision.datasets.ImageFolder(CIFAR100_50_Train_ROOT,
                                                           transform=transforms.Compose([
                                                               transforms.RandomHorizontalFlip(),
                                                               transforms.RandomCrop(size=32, padding=4),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                    std=[0.5, 0.5, 0.5]),
                                                           ])
                                                           )
    CIFAR100_50_test_data = torchvision.datasets.ImageFolder(CIFAR100_50_Test_ROOT,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                   std=[0.5, 0.5, 0.5]),
                                                          ])
                                                          )

    train_data = CIFAR100_50_train_data
    test_data = CIFAR100_50_test_data

    train_dataloader = DataLoader(
        train_data, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    test_dataloader = DataLoader(
        test_data, batch_size=modelConfig["test_batch_size"], shuffle=False, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = UNet(T=modelConfig["T"], num_labels=modelConfig["num_labels"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"])#.to(device)
    print(net_model)
    # load pre-trained Unet
    if modelConfig["training_load_classify_weight"]:
        path = os.path.join(load_classify_dir, f'classify_ckpt_300_.pt')
        load_weight = torch.load(path, map_location='cpu')
        net_model.load_state_dict({k.replace('module.', ''): v for k, v in load_weight.items()})
        print(f'Loading classify_ckpt.....')
    optimizer = torch.optim.AdamW(
            net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["classify_epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["classify_epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    map_dict = read_pkl()
    fc = CosineLinear_PEDCC(latent_variable_dim, class_num)

    # start training
    map_PEDCC = torch.Tensor([])
    for i in range(class_num):
        map_PEDCC = torch.cat((map_PEDCC, map_dict[i].float()), 0)
    map_PEDCC = map_PEDCC.view(class_num, -1)  # (class_num, dimension)
    delta = 0.05
    for e in range(modelConfig["classify_epoch"]):
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_acc = 0

        prev_time = datetime.now()
        l2_norm_trainSample = torch.Tensor([]).cuda()
        with tqdm(train_dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                    optimizer.zero_grad()
                    x_0 = images.cuda()
                    labels = labels.cuda()
                    # diff_labels : for Noneps and eps
                    diff_labels = labels + 1
                    tensor_empty1 = map_PEDCC[labels].float().cuda()
                    label_mse_tensor = tensor_empty1.view(labels.shape[0], -1)  # (b, 256)
                    label_mse_tensor = label_mse_tensor.cuda()
                    if np.random.rand() < 0.1:
                        diff_labels = torch.zeros_like(diff_labels).to(labels.device)

                    # print(diff_labels)
                    noise_mse, feature_vector, t_index = trainer(x_0, diff_labels)  # feature [b 256]

                    output = fc(feature_vector)  # out [b 10]
                    assert output.shape == (x_0.size(0), class_num)

                    with torch.no_grad():
                        l2_norm_trainSample = torch.cat((l2_norm_trainSample, torch.norm(output, p=2, dim=1)),
                                                        dim=0)  # [b]
                    loss2 = NaCLoss(feature_vector, label_mse_tensor, delta, t_index) * 200
                    loss3 = SCLoss(map_PEDCC, labels, feature_vector) * 200  # [class_num 256]  [b]  [b 256]
                    loss1 = noise_mse.sum() / 1000

                    #  print(f'loss1_noise: {loss1}')
                    # assert feature_vector.shape == label_tensor.shape           # [b 1024]
                    # loss2 = F.mse_loss(feature_norm, label_tensor, reduction='none').sum() / 1000.
                    # print(f'loss2_featurre: {loss2}')
                    loss = loss2 + loss3 #+ loss1
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        net_model.parameters(), modelConfig["grad_clip"])
                    optimizer.step()
                    train_loss += loss.item()
                    train_loss1 += loss1.item()
                    train_loss2 += loss2.item()
                    train_loss3 += loss3.item()
                    train_acc += get_cc_ic_PEDCC(output, labels)  # is output
                    # train_acc += get_cc_ic(feature_vector, labels, ui=map_dict, train=True)

        warmUpScheduler.step()
        if e % 100 == 0 and e > 0:
            torch.save(net_model.state_dict(), os.path.join(save_dir, f'classify_ckpt_{e}_.pt'))
        curr_time = datetime.now()
        h, remainder = divmod((curr_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = " Time %02d:%02d:%02d" % (h, m, s)

        # test_dataloader = None
        if test_dataloader is not None:
            valid_loss = 0
            valid_acc = 0
            val_loss1 = 0
            val_loss2 = 0
            val_loss3 = 0
            net_model = net_model.eval()
            trainer = trainer.eval()
            for image, labels in tqdm(test_dataloader, desc="Processing val data: "):
                if torch.cuda.is_available():
                    x_0 = image.cuda()
                    labels = labels.cuda()
                    diff_labels = labels + 1
                    tensor_empty1 = map_PEDCC[labels].float().cuda()
                    label_mse_tensor = tensor_empty1.view(labels.shape[0], -1)  # (b, 256)
                    label_mse_tensor = label_mse_tensor.cuda()

                if np.random.rand() < 0.1:
                    diff_labels = torch.zeros_like(diff_labels).to(labels.device)

                # print(diff_labels)
                noise_mse, feature_vector, t_index = trainer(x_0, diff_labels, False)  # feature [b 256]

                assert feature_vector.shape == (x_0.size(0), latent_variable_dim)
                output = fc(feature_vector)  # out [b 10]
                assert output.shape == (x_0.size(0), class_num)

                with torch.no_grad():
                    l2_norm_trainSample = torch.cat((l2_norm_trainSample, torch.norm(output, p=2, dim=1)),
                                                    dim=0)  # [b]
                    loss2 = NaCLoss(feature_vector, label_mse_tensor, delta, t_index) * 200
                    loss3 = SCLoss(map_PEDCC, labels, feature_vector) * 200  # [class_num 256]  [b]  [b 256]

                    loss1 = noise_mse.sum() / 1000
                # loss2 = F.mse_loss(feature_vector, label_tensor, reduction='none').sum(dim=1)
                # assert loss2.shape == t_index.shape
                # loss2 = loss2 * (1 / (1 + t_index))
                # loss2 = loss2.sum() / 1000. * bias * 100
                # print(f'loss2.shape:{loss2}')
                # print(f'loss2_featurre: {loss2}')
                loss = loss1 + loss2 + loss3

                valid_loss += loss.item()
                val_loss1 += loss1.item()
                val_loss2 += loss2.item()
                val_loss3 += loss3.item()
                # mistake: feature_vector
                valid_acc += get_cc_ic_PEDCC(output, labels)

            epoch_str = ("Epoch %d TrainLoss: %f TrainAcc: %f ValidLoss: %f ValidAcc: %f"
                     % (e, train_loss / len(train_dataloader), train_acc / len(train_dataloader),
                        valid_loss / len(test_dataloader), valid_acc / len(test_dataloader)))
            Loss = ("TrainLoss1: %f TrainLoss2: %f Val_Loss1: %f Val_Loss2: %f"
                    % (train_loss1 / len(train_dataloader), train_loss2 / len(train_dataloader),
                       val_loss1 / len(test_dataloader), val_loss2 / len(test_dataloader)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train.Acc: %f,  "
                         % (e, train_loss / len(train_dataloader), train_acc / len(train_dataloader)))
            Loss = ("Train Loss1: %f, Train Loss2: %f,"
                    % (train_loss1 / len(train_dataloader), train_loss2 / len(train_dataloader)))
        prev_time = curr_time
        f = open(os.path.join(save_dir, f'log_diff_classify_PEDCC_1-{bias}.txt'), 'a+')
        if e == 0:
            f.write(f'begin_time: {curr_time}'.center(100, '=') + '\n')
        print(" ")
        print(epoch_str + time_str)
        print(Loss + ", " + f'LR: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        f.write(epoch_str + time_str + '\n')
        f.write(Loss + ' ' + f'LR: {optimizer.state_dict()["param_groups"][0]["lr"]}' + '\n')
        if e == modelConfig["epoch"] - 1:
            f.write(f'end_time: {curr_time}'.center(100, '=') + '\n')
        f.close()