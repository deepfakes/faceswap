#!/usr/bin/env python3
""" Custom Loss Functions for Faceswap """
import typing as T

from torch import nn

from lib.utils import FaceswapError

from .feature_loss import LPIPSLoss
from .loss import (FocalFrequencyLoss, GeneralizedLoss, GradientLoss,
                   LaplacianPyramidLoss, LInfNorm, LogCosh)
from .flip import LDRFLIPLoss
from .perceptual_loss import GMSDLoss, MSSIMLoss, SSIMLoss


def get_loss_function(name: str, color_order: T.Literal["bgr", "rgb"] = "bgr") -> nn.Module:
    """Get the associated log function for the given configuration file name

    Parameters
    ----------
    name
        The name of the Loss function as specified in the training config file
    color_order
        For flip/lpips only. The color order that the model is training in

    Returns
    -------
    The requested Torch Loss function
    """
    valid = {"ffl": FocalFrequencyLoss,
             "flip": LDRFLIPLoss,
             "gmsd": GMSDLoss,
             "l_inf_norm": LInfNorm,
             "laploss": LaplacianPyramidLoss,
             "logcosh": LogCosh,
             "lpips_alex": LPIPSLoss,
             "lpips_squeeze": LPIPSLoss,
             "lpips_vgg16": LPIPSLoss,
             "ms_ssim": MSSIMLoss,
             "mae": nn.L1Loss,
             "mse": nn.MSELoss,
             "pixel_gradient_diff": GradientLoss,
             "ssim": SSIMLoss,
             "smooth_loss": GeneralizedLoss}
    if name not in valid:
        raise FaceswapError(f"'{name}' is not a valid Loss function. Choose from: {list(valid)}")

    kwargs: dict[str, T.Any] = {}
    if name in ("mae", "mse"):
        kwargs["reduction"] = "none"
    if name == "flip" or name.startswith("lpips_"):
        kwargs["color_order"] = color_order
    if name.startswith("lpips_"):
        kwargs["trunk_network"] = name.rsplit("_", maxsplit=1)[-1]
        kwargs["crop"] = True
    return valid[name](**kwargs)
