from configs.diffusion import *
from configs.unet import *

outer_resnet_config = ResNetConfig(
    in_channels=-1,
    out_channels=-1,
    num_groups_norm=32,
    dropout=0.0,
    use_attention_ffn=False
)

inner_resnet_config = ResNetConfig(
    in_channels=-1,
    out_channels=-1,
    num_groups_norm=32,
    dropout=0.0,
    use_attention_ffn=True
)

inner_unet_config = UNetConfig(
    resnet_config=inner_resnet_config,
    time_emb_dim=128,  # Значение по умолчанию, т.к. temporal_dim=null в inner_config
    resolution_channels=[128, 192],
    num_resnets_per_resolution=[2, 2],
    num_attention_layers=[1, 1],
    conditioning_feature_dim=-1,
    conditioning_feature_proj_dim=-1,
    skip_mid_blocks=False,
    nesting=True
)

nested_unet_config = NestedUNetConfig(
    resnet_config=outer_resnet_config,
    time_emb_dim=1024,
    resolution_channels=[64, 128],
    num_resnets_per_resolution=[2, 1],
    num_attention_layers=[0, 0],
    conditioning_feature_dim=-1,
    conditioning_feature_proj_dim=-1,
    skip_mid_blocks=True,
    nesting=False,
    inner_config=inner_unet_config,
    skip_inner_unet_input=False,
    skip_normalization=True,
    freeze_inner_unet=False,
    interp_conditioning=False
)


diffusion_config = DiffusionConfig(
    num_diffusion_steps=1000,
    schedule_type=ScheduleType.DDPM,
    prediction_type=PredictionType.V_PREDICTION,
    use_double_loss=True,
    multi_res_weights=[4.0, 1.0]
)
