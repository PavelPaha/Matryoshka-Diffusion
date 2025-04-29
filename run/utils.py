
import os
from configs.diffusion import DiffusionConfig
from configs.unet import NestedUNetConfig
from models.diffusion import NestedDiffusion
from models.nested_unet import NestedUNet


from torch.utils.tensorboard import SummaryWriter



def get_logger(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logger = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    return logger


def get_model(unet_config: NestedUNetConfig, diffusion_config: DiffusionConfig, device):
    unet_model = NestedUNet(
        input_channels=3,
        output_channels=3,
        config=unet_config,
    ).to(device)
    return NestedDiffusion(vision_model=unet_model, config=diffusion_config).to(device)
