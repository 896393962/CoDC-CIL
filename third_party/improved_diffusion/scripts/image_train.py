"""
Train a diffusion model on images.
"""

import argparse

import torch

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop


device = "cuda: 0" if torch.cuda.is_available() else'cpu'

def main():
    print(torch.__version__)
    # print(torchvision.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.is_available())
    # 初始默认配置
    args = create_argparser().parse_args()


    dist_util.setup_dist()
    logger.configure('../logs_test')
    logger.log(str(args))

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    logger.log(model)
    model.to(dist_util.dev())
    # 采样器 schedule_sampler  时间t
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,             # adam优化器参数
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    # 通用默认args
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=8,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    MODEL_FLAGS = "--image_size 64 --num_channels 128 --num_res_blocks 3"
    DIFFUSION_FLAGS = "--diffusion_steps 4000 --noise_schedule linear"
    TRAIN_FLAGS = "--lr 1e-4 --batch_size 128"
    # train
    # python scripts/image_train.py --data_dir path/to/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

    # sample

    '''
    python scripts/image_sample.py --model_path /path/to/model.pt $MODEL_FLAGS $DIFFUSION_FLAGS --use_ddim True
    --timestep_respacing ddim50 --sample_name='...'
    python scripts/image_sample.py --model_path /path/to/model.pt $MODEL_FLAGS $DIFFUSION_FLAGS --use_ddim False
    --sample_name='DDPM'
    python scripts/image_sample.py --model_path /path/to/model.pt $MODEL_FLAGS $DIFFUSION_FLAGS --use_ddim False
    --timestep_respacing 250 --sample_name='...'
    '''
    main()
