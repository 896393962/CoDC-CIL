import os

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from Diffusion.Train import latent_variable_dim, save_dir, class_num
import matplotlib.pyplot as plt
import pickle,torch
import numpy as np
import cv2
from torchvision import transforms
from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.ModelCondition import UNet
import logger

data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

def gen_data(mean,cov,num):
    mean=mean.cpu()
    data = np.random.multivariate_normal(mean, cov, num)
    # 生成一个多元正态分布矩阵
    data = np.random.multivariate_normal(mean, cov, num)
    # data = np.vstack(np.tile(mean, (num, 1)))  # only mean sample
    # 保留四位有效数字
    return np.round(data, 4)

def save_map_mean_var(subset, load_batch, trainer, train_root="./dataset/cifar100/train"):
    map_mean_var = {}
    if not os.path.isdir(train_root):
        raise FileNotFoundError(
            f"Dataset root does not exist: {train_root}. "
            "Set pedcc_train_root in Main.py before generating PEDCC statistics."
        )
    # train_root=r"./data/MNIST_img/train"

    '''
    the images are arranged in this way: ::

    root/0/0/xxx.png
    root/0/0/xxy.png
    root/0/0/xxz.png

    root/1/1/123.png
    root/1/1/nsdf3.png
    root/1/1/asd932_.png

    '''
    folder_list = os.listdir(train_root)
    for i in sorted(folder_list):
        train_folder = os.path.join(train_root, i)  # single class directory
        data_set = ImageFolder(train_folder, transform=data_transform)
        # input images on single class directory
        data_data = DataLoader(data_set, batch_size=load_batch, shuffle=False)
        out_all = torch.Tensor([]).cuda()
        #  the dimention of predefined class centriods
        out_sum = torch.Tensor([0] * latent_variable_dim).cuda()
        with torch.no_grad():
            for im, labels in data_data:
                # print((im[:, 0, :, :] == im[:, 1, :, :]).sum().item())
                im = im.cuda()
                labels = labels.cuda()
                # no noise_level images for calc every class mean and cov
                noise_mse, feature_vector, _ = trainer(im, labels)  # feature_vector [b 1024]
                assert feature_vector.size(1) == 1024  # PEDCC dim: 1024
                out_all = torch.cat((out_all, feature_vector), 0)
                for single_feature in feature_vector:
                    out_sum += single_feature
        print(i)
        mean_out = out_sum / out_all.size(0)
        cov = np.cov(torch.t(out_all).cpu().numpy())  # 矩阵转置，协方差
        map_mean_var[mean_out] = cov

    f = open(subset, 'wb')
    pickle.dump(map_mean_var, f)
    f.close()
    print(f'is saving mean_var.pkl')

def PEDCC_sampler(modelConfig):
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        train_root = modelConfig.get("pedcc_train_root", modelConfig.get("train_root", "./dataset/cifar100/train"))
        if not os.path.isdir(train_root):
            raise FileNotFoundError(
                f"Dataset root does not exist: {train_root}. "
                "Set pedcc_train_root in Main.py or pass it through modelConfig."
            )
        net_model = UNet(T=modelConfig["T"], num_labels=modelConfig["num_labels"], ch=modelConfig["channel"],
                         ch_mult=modelConfig["channel_mult"], num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"])

        save_dir = os.path.join('./Checkpoints/unPEDCC_POD/CIL_EMNIST',
                                f'{20}_bias_1-{300}_FC(Augm(0.5)_X0_t=Tstep_Cemb_TwightPOD_labels')
        # save_dir = os.path.join('./Checkpoints/unPEDCC_POD/CIFAR100',
        #                         f'fleezeLeft_down-1_bias_1-{300}_FC(Augm(0.5)_X0_t=Tstep_Cemb_TwightPOD_labels')
        ckpt = torch.load(os.path.join(
            save_dir, modelConfig["test_load_weight"]), map_location=device)
        net_model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
        print(f"UNet_model: {modelConfig['test_load_weight']} load weight done.")
        net_model.eval()

        logger.configure(save_dir)
        sampler = GaussianDiffusionSampler(
            net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        data_std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
        data_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
        data_num = 100
        saveImg_dir_true = os.path.join(save_dir, 'diffgen_true')
        saveImg_dir_false = os.path.join(save_dir, 'diffgen_false')
        for i in range(10):
            class_num = i + 10   # which class; 1 --> the first class: plane(CIFAR10)
            if not os.path.exists(os.path.join(saveImg_dir_true, f'{class_num:02}')):
                os.makedirs(os.path.join(saveImg_dir_true, f'{class_num:02}'))
            saveImg_path_true = os.path.join(saveImg_dir_true, f'{class_num:02}')
            if not os.path.exists(os.path.join(saveImg_dir_false, f'{class_num:02}')):
                os.makedirs(os.path.join(saveImg_dir_false, f'{class_num:02}'))
            saveImg_path_false = os.path.join(saveImg_dir_false, f'{class_num:02}')
            true_count = 0
            false_count = 0
            loop = 0
            try:
                while true_count < 2400:
                    noisyImage = torch.randn(
                        size=[data_num, 3, 32, 32], device=device)
                    labels = torch.randint(class_num + 1, class_num + 2, size=(data_num,), device=device)
                    sampledImg, pred_label = sampler(noisyImage, PEDCC_feature_vector=None, labels=labels)
                    logger.log(pred_label)
                    true_false = (labels == pred_label + 1).tolist()
                    # sampledImgs = sampledImg * 0.5 + 0.5  # [0 ~ 1]
                    sampledImgs = sampledImg * data_std + data_mean  # [0 ~ 1] data process
                    for k in range(sampledImgs.size(0)):
                        print(f'sampledImgs_{class_num}_{k + loop * data_num}, classify: {true_false[k]}')
                        if true_false[k]:
                            save_image(sampledImgs[k], os.path.join(saveImg_path_true,
                            f'{class_num}_True_e=600_{k + loop * data_num}.png'), nrow=1, padding=0)
                            true_count = true_count + 1
                            if true_count >= 2400 or loop >= 30:
                                raise StopIteration
                        else:
                            save_image(sampledImgs[k], os.path.join(saveImg_path_false,
                            f'{class_num}_false_e=600_{k + loop * data_num}.png'),nrow=1, padding=0)
                            false_count = false_count + 1
                    loop += 1
            except StopIteration:
                pass
            genRate = true_count / (true_count + false_count)
            logger.log(f'class: {class_num}, T={true_count}, '
                  f'F={false_count}, genRate={genRate:.4f}')
