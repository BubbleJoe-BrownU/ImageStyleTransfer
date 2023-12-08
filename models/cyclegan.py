import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class CycleGANConfig:
    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    ndf: int = 64
    netG: str = 'resnet_9blocks'
    netD: str = 'basic'
    n_layers_D: int = 3
    norm: str = 'instance'
    init_type: str = 'normal'
    init_gain: float = 0.02
    no_dropout: bool = False
    no_lsgan: bool = False
    pool_size: int = 0
    lr: float = 0.0002
    beta1: float = 0.5
    lr_policy: str = 'linear'
    lr_decay_iters: int = 50
    epoch_count: int = 1
    niter: int = 100
    niter_decay: int = 100
    save_epoch_freq: int = 5
    save_latest_freq: int = 5000
    print_freq: int = 100
    continue_train: bool = False

def make_generator(config):
    """Create a generator
    We provide two choices of generators: resnet-based generators and unet-based generators.
    Resnet-based generators consists of several resnet blocks between a few downsampling/upsampling operations.
    Unet-based generators consists of U-Net encoder-decoder structure with skip connections between mirrored layers in the encoder and decoder stacks.
    """
    return None

def make_discriminator(config):
    """
    We implemented a 1x1 PatchGAN discriminator here
    """
    return nn.Sequential(
        nn.Conv2d(config.input_nc, config.ndf, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(config.ndf, config.ndf * 2, kernel_size=1, stride=1, padding=0),
        nn.InstanceNorm2d(config.ndf * 2),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(config.ndf * 2, 1, kernel_size=1, stride=1, padding=0)
    )
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, out_nc, in_nc):


class UnetGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.

class CycleGAN(nn.Module):
    def __init__(self, config: CycleGANConfig):
        super().__init__(self, config)
        self.