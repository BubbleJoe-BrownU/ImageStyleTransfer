import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import functools

from transformers import Trainer, TrainingArguments




@dataclass
class UNetConfig:
    num_downs: int = 8 # number of downsample/upsample layers in UNet
    ngf: int = 64 # number of filters in the last conv layer
    norm_layer = nn.BatchNorm2d
    use_dropout: bool = False
    input_nc: int = 3
    output_nc: int = 3


class UNetSkipConnectionBlock(nn.Module):
    """
    This class implements the skip connection block used in the UNet architecture
    In -------------------identity----------------- Out
    | ---downsample----| submodules | ---upsample--- |
    The submodule can be a UNetSkipConnectionBlock
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        if type(norm) == functools.partial:
            use_bias = norm.func == nn.InstanceNorm2d
        else:
            use_bias = norm == nn.InstanceNorm2d
        downsample = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm(outer_nc)

        if innermost:
            # no submodule
            upsample = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downsample]
            up = [uprelu, upsample, upnorm]
            model = down + up
        elif outermost:
            upsample = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downsample]
            up = [uprelu, upsample, nn.Tanh()]
            model = down + [submodule] + up
        else:
            upsample = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downsample, downnorm]
            up = [uprelu, upsample, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            # add skip connections
            return torch.cat([x, self.model(x)], 1)



class UNet(nn.Module):
    def __init__(self, UNetConfig):
        super().__init__()
        self.config = UNetConfig
        # create the innermost UNet block
        unet_block = UNetSkipConnectionBlock(self.config.ngf * 8, self.config.ngf * 8, input_nc=None, submodule=None, innermost=True, norm=self.config.norm_layer, use_dropout=self.config.use_dropout)
        for i in range(self.config.num_downs - 5):
            unet_block = UNetSkipConnectionBlock(self.config.ngf * 8, self.config.ngf * 8, input_nc=None, submodule=unet_block, norm=self.config.norm_layer, use_dropout=self.config.use_dropout)
        unet_block = UNetSkipConnectionBlock(self.config.ngf * 4, self.config.ngf * 8, input_nc=None, submodule=unet_block, norm=self.config.norm_layer)
        unet_block = UNetSkipConnectionBlock(self.config.ngf * 2, self.config.ngf * 4, input_nc=None, submodule=unet_block, norm=self.config.norm_layer)
        unet_block = UNetSkipConnectionBlock(self.config.ngf, self.config.ngf * 2, input_nc=None, submodule=unet_block, norm=self.config.norm_layer)
        self.model = UNetSkipConnectionBlock(self.config.output_nc, self.config.ngf, input_nc=self.config.input_nc, submodule=unet_block, outermost=True, norm=self.config.norm_layer)

    def forward(self, x, label=None):
        output = self.model(x)
        if label is not None:
            loss = F.kl_div(output, label) + F.mse_loss(output, label)
        return output if label is None else (output, loss)
    
    def calc_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    


if __name__ == "__main__":
    # as a sanity check, or simply for fun, running this script will train an UNet AutoEncoder on CIFAR10
    config = UNetConfig()
    model = UNet(config)

    args = TrainingArguments(
        output_dir="../unet_ae_results",
        num_train_epochs=10,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=None,
        eval_dataset=None,
        compute_metrics=None
    )
    print(model(x).shape)
