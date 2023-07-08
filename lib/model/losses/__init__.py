#!/usr/bin/env python3
""" Custom Loss Functions for Faceswap """

from .feature_loss import LPIPSLoss
from .loss import (FocalFrequencyLoss, GeneralizedLoss, GradientLoss,
                   LaplacianPyramidLoss, LInfNorm, LossWrapper)
from .perceptual_loss import DSSIMObjective, GMSDLoss, LDRFLIPLoss, MSSIMLoss
