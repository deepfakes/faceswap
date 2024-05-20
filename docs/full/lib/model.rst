*************
model package
*************

The Model Package handles interfacing with the neural network backend and holds custom objects.

.. contents:: Contents
   :local:

model.backup_restore module
===========================

.. automodule:: lib.model.backup_restore
   :members:
   :undoc-members:
   :show-inheritance:

model.initializers module
=========================

.. rubric:: Module Summary

.. autosummary::
   :nosignatures:

   ~lib.model.initializers.ConvolutionAware
   ~lib.model.initializers.ICNR
   ~lib.model.initializers.compute_fans

.. automodule:: lib.model.initializers
   :members:
   :undoc-members:
   :show-inheritance:

model.layers module
===================

.. rubric:: Module Summary

.. autosummary::
   :nosignatures:

   ~lib.model.layers.GlobalMinPooling2D
   ~lib.model.layers.GlobalStdDevPooling2D
   ~lib.model.layers.KResizeImages
   ~lib.model.layers.L2_normalize
   ~lib.model.layers.PixelShuffler
   ~lib.model.layers.QuickGELU
   ~lib.model.layers.ReflectionPadding2D
   ~lib.model.layers.SubPixelUpscaling
   ~lib.model.layers.Swish

.. automodule:: lib.model.layers
   :members:
   :undoc-members:
   :show-inheritance:

model.losses module
===================

.. rubric:: Module Summary

.. autosummary::
   :nosignatures:

   ~lib.model.losses.loss.FocalFrequencyLoss
   ~lib.model.losses.loss.GeneralizedLoss
   ~lib.model.losses.loss.GradientLoss
   ~lib.model.losses.loss.LaplacianPyramidLoss
   ~lib.model.losses.loss.LInfNorm
   ~lib.model.losses.loss.LossWrapper
   ~lib.model.losses.feature_loss.LPIPSLoss
   ~lib.model.losses.loss.perceptual_loss.DSSIMObjective
   ~lib.model.losses.loss.perceptual_loss.GMSDLoss
   ~lib.model.losses.loss.perceptual_loss.LDRFLIPLoss
   ~lib.model.losses.loss.perceptual_loss.MSSIMLoss

.. automodule:: lib.model.loss.losses.loss
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: lib.model.loss.feature_loss
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: lib.model.loss.perceptual_loss
   :members:
   :undoc-members:
   :show-inheritance:


model.nets module
=================

.. rubric:: Module Summary

.. autosummary::
   :nosignatures:

   ~lib.model.nets.AlexNet
   ~lib.model.nets.SqueezeNet

.. automodule:: lib.model.nets
   :members:
   :undoc-members:
   :show-inheritance:

model.nn_blocks module
======================

.. rubric:: Module Summary

.. autosummary::
   :nosignatures:

   ~lib.model.nn_blocks.Conv2D
   ~lib.model.nn_blocks.Conv2DBlock
   ~lib.model.nn_blocks.Conv2DOutput
   ~lib.model.nn_blocks.ResidualBlock
   ~lib.model.nn_blocks.SeparableConv2DBlock
   ~lib.model.nn_blocks.Upscale2xBlock
   ~lib.model.nn_blocks.UpscaleBlock
   ~lib.model.nn_blocks.set_config

.. automodule:: lib.model.nn_blocks
   :members:
   :undoc-members:
   :show-inheritance:

model.normalization module
==========================

.. rubric:: Module Summary

.. autosummary::
   :nosignatures:

   ~lib.model.normalization.InstanceNormalization

.. automodule:: lib.model.normalization
   :members:
   :undoc-members:
   :show-inheritance:

model.optimizers module
=======================

.. rubric:: Module Summary

.. autosummary::
   :nosignatures:

   ~lib.model.optimizers.AdaBelief

.. automodule:: lib.model.optimizers
   :members:
   :undoc-members:
   :show-inheritance:

model.session module
=====================

.. automodule:: lib.model.session
   :members:
   :undoc-members:
   :show-inheritance:
