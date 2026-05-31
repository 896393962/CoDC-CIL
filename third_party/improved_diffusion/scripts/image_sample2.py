"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import time

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image
from torchvision.utils import save_image

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

CURRENT_NUM_CLASSES = 20

def main():
    args = create_argparser().parse_args()
    print(args)

    dist_util.setup_dist()
    dir = './logs_4/sample'
    logger.configure(dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    start_time = time.time()
    all_images = []
    all_labels = []
    num_per_class = 500
    batch_class_index = 18
    low_class = batch_class_index
    high_class = low_class + 1
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            if len(all_images) > 0 and len(all_images) * args.batch_size % num_per_class == 0:
                low_class = low_class + 1
                high_class = high_class + 1
            classes = th.randint(
                low=low_class, high=high_class, size=(args.batch_size,), device=dist_util.dev()
            )
            print(classes)
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        # print(sample.shape)
        # grid image save
        # save_image(sample, os.path.join(dir, '0.png'), nrow=8)
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        gen_dir = os.path.join(logger.get_dir(), 'diffgen_class_imgs')
        if not os.path.isdir(gen_dir):
            os.makedirs(gen_dir)
        for i in range(args.num_samples):
            im = Image.fromarray(arr[i])
            class_path = os.path.join(gen_dir, f'{args.timestep_respacing}-{batch_class_index+i//num_per_class:02d}')
            if not os.path.isdir(class_path):
                os.makedirs(class_path)
            im.save(f'{class_path}/{batch_class_index+i//num_per_class:02d}_diffgen_{i}.png')
            if (i+1) % num_per_class == 0 and i > 0:
                logger.log(f"saving {i}th image to {class_path}")
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{args.timestep_respacing}-{batch_class_index:02d}-"
                                                  f"{batch_class_index+i//num_per_class:02d}_{shape_str}.npz")
        logger.log(f"saving npz to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    # images = np.load(out_path)['arr_0']
    # for i in range (16):
    #     images = Image.fromarray(images[0])
    #     images.save(os.path.join(dir, f'{i}.png'))
    # labels = np.load(out_path)['arr_1']
    used_time = time.time() - start_time
    timeArray = time.localtime(time.time())
    StyleTime = time.strftime('%Y.%m.%d %H:%M:%S', timeArray)
    logger.log(f"sampling {args.num_samples} complete, used time {used_time:.2f} seconds, Current Time: {StyleTime}")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1000,
        batch_size=100,
        use_ddim=False,
        model_path="./logs_4/ema_0.9999_100000.pt",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
