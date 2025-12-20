*************
train package
*************

The Train Package handles the Model and Trainer plugins for training models in Faceswap.

.. contents:: Contents
   :local:

model package
=============

This package contains various helper functions that plugins can inherit from

.. automodapi:: plugins.train.model._base.inference
   :include-all-objects:
   :no-inheritance-diagram:

.. automodapi:: plugins.train.model._base.io
   :include-all-objects:
   :no-inheritance-diagram:

|
.. automodapi:: plugins.train.model._base.model
   :include-all-objects:
   :no-inheritance-diagram:

|
.. automodapi:: plugins.train.model._base.settings
   :include-all-objects:
   :no-inheritance-diagram:

|
.. automodapi:: plugins.train.model._base.state
   :include-all-objects:
   :no-inheritance-diagram:

|
.. automodapi:: plugins.train.model._base.update
   :include-all-objects:
   :no-inheritance-diagram:

|
.. automodapi:: plugins.train.model.original
   :include-all-objects:


trainer package
===============

This package contains the training loop for Faceswap

.. automodapi:: plugins.train.trainer._base
   :include-all-objects:
   :no-inheritance-diagram:

|
.. automodapi:: plugins.train.trainer._display
   :include-all-objects:
   :no-inheritance-diagram:

|
.. automodapi:: plugins.train.trainer.distributed
   :include-all-objects:

|
.. automodapi:: plugins.train.trainer.original
   :include-all-objects:

|
.. automodapi:: plugins.train.trainer.trainer_config
   :include-all-objects:
