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

.. rubric:: Module Summary

.. autosummary::
   :nosignatures:

   ~lib.model.losses.DSSIMObjective
   ~lib.model.losses.PenalizedLoss
   ~lib.model.losses.GeneralizedLoss
   ~lib.model.losses.GMSDLoss
   ~lib.model.losses.GradientLoss
   ~lib.model.losses.LInfNorm

.. automodule:: lib.model.losses
   :members:
   :undoc-members:
   :show-inheritance:

model.nn_blocks module
----------------------

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

