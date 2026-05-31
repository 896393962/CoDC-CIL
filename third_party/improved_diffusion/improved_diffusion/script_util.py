import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import SuperResModel, UNetModel

NUM_CLASSES = 100


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )


def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,       # 模型是否使用固定方差或者预测方差
    sigma_small,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,  # 那些需要attention
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    print(model)
    return model, diffusion


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    if image_size == 128:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
        '''
        data_dir='F:\\dataset\\CIFAR10\\CIFAR10\\train', schedule_sampler='uniform', lr=0.0001, weight_decay=0.0, lr_anneal_steps=0, 
        batch_size=16, microbatch=-1, ema_rate='0.9999', log_interval=10, save_interval=10000, resume_checkpoint='', use_fp16=False, 
        fp16_scale_growth=0.001, image_size=32, num_channels=128, num_res_blocks=3, num_heads=4, num_heads_upsample=-1, 
        attention_resolutions='16,8', dropout=0.3, learn_sigma=True, sigma_small=False, class_cond=True, diffusion_steps=4000, 
        noise_schedule='cosine', timestep_respacing='', use_kl=False, predict_xstart=False, rescale_timesteps=True, 
        rescale_learned_sigmas=True, use_checkpoint=False, use_scale_shift_norm=True)
        '''

    return UNetModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


def sr_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 256
    res["small_size"] = 64
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def sr_create_model_and_diffusion(
    large_size,
    small_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
):
    model = sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def sr_create_model(
    large_size,
    small_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    _ = small_size  # hack to prevent unused variable

    if large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {large_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    return SuperResModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    #　生成一个扩散过程的框架
    # beta 余弦方案
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    print(betas)
    '''  前100
    [1.e-05, 1.e-05, 1.e-05, 1.e-05, 1.e-05, 1.e-05, 1.e-05, 1.e-05, 1.e-05, 1.e-05, 1.e-05, 1.e-05, 1.e-05, 1.e-05, 1.e-05, 1.e-05,
       1.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05,
    2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05, 2.e-05,
    2.e-05, 2.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05,
    3.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05, 3.e-05,
    3.e-05, 3.e-05, 3.e-05, 4.e-05, 4.e-05, 4.e-05, 4.e-05, 4.e-05, 4.e-05, 4.e-05, 4.e-05, 4.e-05, 4.e-05, 4.e-05, 4.e-05, 4.e-05,
    4.e-05, 4.e-05, 4.e-05, 4.e-05]
    [0.01989, 0.02009, 0.02029, 0.0205 , 0.02072, 0.02093, 0.02115,0.02138, 0.02161, 0.02185, 0.02209, 0.02234, 0.02259, 0.02285,
    0.02311, 0.02338, 0.02366, 0.02394, 0.02423, 0.02453, 0.02484,0.02515, 0.02547, 0.0258 , 0.02614, 0.02648, 0.02684, 0.0272 ,
    0.02758, 0.02796, 0.02836, 0.02877, 0.02919, 0.02962, 0.03007, 0.03053, 0.031  , 0.03149, 0.03199, 0.03251, 0.03305, 0.03361,
    0.03418, 0.03477, 0.03539, 0.03603, 0.03669, 0.03737, 0.03809,0.03883, 0.0396 , 0.0404 , 0.04123, 0.0421 , 0.043  , 0.04395,
    0.04493, 0.04597, 0.04705, 0.04818, 0.04937, 0.05062, 0.05194,0.05332, 0.05478, 0.05632, 0.05796, 0.05968, 0.06152, 0.06347,
    0.06555, 0.06777, 0.07015, 0.0727 , 0.07544, 0.0784 , 0.0816 ,0.08506, 0.08884, 0.09297, 0.0975 , 0.10249, 0.10802, 0.11419,
    0.12109, 0.12889, 0.13775, 0.14793, 0.15972, 0.17355, 0.19   ,0.20988, 0.23437, 0.26531, 0.30556, 0.36   , 0.4375 , 0.55556,
    0.75   , 0.999  ])

    '''
    if use_kl:
        # print('KL-loss')
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        # 默认True 根据learn_sigma
        # print('rescale_learned_sigmas')
        loss_type = gd.LossType.RESCALED_MSE
    else:
        # print('MSE')
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        # print(f'timestep_respacing: {[steps]}')
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),  # {0 1 2 ... 1000}
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
