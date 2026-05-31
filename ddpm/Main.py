from Diffusion.Train import train, classify
from PEDCC.gen_PEDCC_sampler import PEDCC_sampler


def main(model_config = None):
    modelConfig = {
        "state": "train",  # train
        # "state": "classifier",  #  eval
        # "state": "PEDCC_sampler",   # PEDCC_sampler
        "epoch": 1001,
        "classify_epoch": 401,
        "batch_size": 32,
        "test_batch_size": 8,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 4, 4],
        "attn": [2],
        "num_labels": 40,    # ddpm cond = 100 fixed
        "num_res_blocks": 3,
        "dropout": 0.15,
        "lr": 1e-4,          # 1e-4
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "train_root": "./dataset/cifar100/train",
        "test_root": "./dataset/cifar100/test",
        "classify_train_root": "./dataset/cifar100_50/train",
        "classify_test_root": "./dataset/cifar100_50/test",
        "pedcc_train_root": "./dataset/cifar100/train",
        "training_load_classify_weight": False,
        "training_load_generator_weight": False,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_600_.pt",
        "sampled_dir": "./CIL_CIFAR100/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "800_8imgs_class(1).png",
        "nrow": 8
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    elif modelConfig["state"] == "PEDCC_sampler":
        PEDCC_sampler(modelConfig)
    else:
        classify(modelConfig)


if __name__ == '__main__':
    main()
