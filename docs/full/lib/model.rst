model package
=============

The Model Package handles interfacing with the neural network backend and holds custom objects.

.. contents:: Contents
   :local:

model.backup_restore module
---------------------------

.. automodule:: lib.model.backup_restore
   :members:
   :undoc-members:
   :show-inheritance:

model.initializers module
-------------------------

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
-------------------

.. rubric:: Module Summary

.. autosummary::
   :nosignatures:
   
   ~lib.model.layers.GlobalMinPooling2D
   ~lib.model.layers.GlobalStdDevPooling2D
   ~lib.model.layers.L2_normalize
   ~lib.model.layers.PixelShuffler
   ~lib.model.layers.ReflectionPadding2D
   ~lib.model.layers.SubPixelUpscaling
   
.. automodule:: lib.model.layers
   :members:
   :undoc-members:
   :show-inheritance:

model.losses module
-------------------

The losses listed here are generated from the docstrings in :mod:`lib.model.losses_tf`, however
the functions are excactly the same for :mod:`lib.model.losses_plaid`. The correct loss module will
be imported as :mod:`lib.model.losses` depending on the backend in use.

.. rubric:: Module Summary

.. autosummary::
   :nosignatures:

   ~lib.model.losses_tf.DSSIMObjective
   ~lib.model.losses_tf.PenalizedLoss
   ~lib.model.losses_tf.GeneralizedLoss
   ~lib.model.losses_tf.GMSDLoss
   ~lib.model.losses_tf.GradientLoss
   ~lib.model.losses_tf.LInfNorm

.. automodule:: lib.model.losses_tf
   :members:
   :undoc-members:
   :show-inheritance:

model.nn_blocks module
----------------------

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
--------------------------

.. rubric:: Module Summary

.. autosummary::
   :nosignatures:
   
   ~lib.model.normalization.InstanceNormalization
   
.. automodule:: lib.model.normalization
   :members:
   :undoc-members:
   :show-inheritance:

model.session module
---------------------

.. automodule:: lib.model.session
   :members:
   :undoc-members:
   :show-inheritance:

