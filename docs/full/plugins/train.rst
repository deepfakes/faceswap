*************
train package
*************

The Train Package handles the Model and Trainer plugins for training models in Faceswap.


.. contents:: Contents
   :local:

model package
=============

This package contains various helper functions that plugins can inherit from

.. rubric:: Module Summary

.. autosummary::
   :nosignatures:

   ~plugins.train.model._base.model
   ~plugins.train.model._base.settings
   ~plugins.train.model._base.io

model._base.model module
------------------------

.. automodule:: plugins.train.model._base.model
   :members:
   :undoc-members:
   :show-inheritance:

model._base.settings module
---------------------------

.. automodule:: plugins.train.model._base.settings
   :members:
   :undoc-members:
   :show-inheritance:

model._base.io module
---------------------

.. automodule:: plugins.train.model._base.io
   :members:
   :undoc-members:
   :show-inheritance:

model.original module
----------------------

.. automodule:: plugins.train.model.original
   :members:
   :undoc-members:
   :show-inheritance:

trainer package
===============

This package contains the training loop for Faceswap

.. rubric:: Module Summary

.. autosummary::
   :nosignatures:

   ~plugins.train.trainer._base
   ~plugins.train.model._display
   ~plugins.train.model._tensorboard

trainer._base module
----------------------

.. automodule:: plugins.train.trainer._base
   :members:
   :undoc-members:
   :show-inheritance:

trainer._display module
-----------------------

.. automodule:: plugins.train.trainer._display
   :members:
   :undoc-members:
   :show-inheritance:

trainer._tensorboard module
---------------------------

.. automodule:: plugins.train.trainer._tensorboard
   :members:
   :undoc-members:
   :show-inheritance:
