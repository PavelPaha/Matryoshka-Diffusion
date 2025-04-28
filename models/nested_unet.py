import torch
import torch.nn as nn

from configs.unet import NestedUNetConfig
from models.unet import UNet
from models.utils import zero_module


class NestedUNet(UNet):
    def __init__(self, input_channels, output_channels, config: NestedUNetConfig):
        super().__init__(input_channels, output_channels=output_channels, config=config)
        config.inner_config.conditioning_feature_dim = config.conditioning_feature_dim
        if (
            getattr(config.inner_config, "inner_config", None) is None
        ):  # основной мув с вложенностью
            self.inner_unet = UNet(input_channels, output_channels, config.inner_config)
        else:
            self.inner_unet = NestedUNet(
                input_channels, output_channels, config.inner_config
            )

        if not config.skip_inner_unet_input:
            self.in_adapter = zero_module(
                nn.Conv2d(
                    config.resolution_channels[-1],
                    config.inner_config.resolution_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
            )
        else:
            self.in_adapter = None

        self.out_adapter = zero_module(
            nn.Conv2d(
                config.inner_config.resolution_channels[0],
                config.resolution_channels[-1],
                kernel_size=3,
                padding=1,
                bias=True,
            )
        )

        nest_ratio = int(2 ** (len(config.resolution_channels) - 1))
        if (
            self.inner_unet.config.nesting
            and self.inner_unet.model_type == "nested_unet"
        ):
            self.nest_ratio = [
                nest_ratio * self.inner_unet.nest_ratio[0]
            ] + self.inner_unet.nest_ratio
        else:
            self.nest_ratio = [nest_ratio]

        if config.freeze_inner_unet:
            for p in self.inner_unet.parameters():
                p.requires_grad = False
        if config.interp_conditioning:
            self.interp_layer1 = nn.Linear(self.time_emb_dim // 4, self.time_emb_dim)
            self.interp_layer2 = nn.Linear(self.time_emb_dim, self.time_emb_dim)

    def forward_conditioning(self, *args, **kwargs):
        return self.inner_unet.forward_conditioning(*args, **kwargs)

    def forward_denoising(self, x_t, times, conditioning=None, cond_mask=None):
        # 1. time embedding
        temb = self.create_time_emb(times)

        # 2. input layer (normalize the input)
        if self.config.nesting:
            x_t, x_feat = x_t
        bsz = [x.size(0) for x in x_t]
        bh, bl = bsz[0], bsz[1]
        x_t_low, x_t = x_t[1:], x_t[0]
        x = self.forward_input_layer(
            x_t, normalize=(not self.config.skip_normalization)
        )
        if self.config.nesting:
            x = x + x_feat

        # 3. downsample blocks in the outer layers
        x, skip_activations = self.forward_downsample(
            x,
            temb[:bh],
            conditioning[:bh],
            cond_mask[:bh] if cond_mask is not None else cond_mask,
        )

        # 4. run inner unet
        x_inner = self.in_adapter(x) if self.in_adapter is not None else None
        x_inner = (
            torch.cat([x_inner, x_inner.new_zeros(bl - bh, *x_inner.size()[1:])], 0)
            if bh < bl
            else x_inner
        )  # pad zeros for low-resolutions
        x_low, x_inner = self.inner_unet.forward_denoising(
            (x_t_low, x_inner), times, conditioning, cond_mask
        )
        x_inner = self.out_adapter(x_inner)
        x = x + x_inner[:bh] if bh < bl else x + x_inner

        # 5. upsample blocks in the outer layers
        x = self.forward_upsample(
            x,
            temb[:bh],
            conditioning[:bh],
            cond_mask[:bh] if cond_mask is not None else cond_mask,
            skip_activations,
        )

        # 6. output layer
        x_out = self.forward_output_layer(x)

        # 7. outpupt both low and high-res output
        if isinstance(x_low, list):
            out = [x_out] + x_low
        else:
            out = [x_out, x_low]
        if self.config.nesting:
            return out, x
        return out

    @property
    def model_type(self):
        return "nested_unet"
