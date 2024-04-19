*************
align package
*************

The align Package handles detected faces, their alignments and masks.

.. contents:: Contents
   :local:


aligned\_face module
====================

Handles aligned faces and corresponding pose estimates

.. rubric:: Module Summary

.. autosummary::
   :nosignatures:

   ~lib.align.aligned_face.AlignedFace
   ~lib.align.aligned_face.get_matrix_scaling
   ~lib.align.aligned_face.transform_image

.. rubric:: Module

.. automodule:: lib.align.aligned_face
   :members:
   :undoc-members:
   :show-inheritance:


aligned\_mask module
====================

Handles aligned storage and retrieval of Faceswap generated masks

.. rubric:: Module Summary

.. autosummary::
   :nosignatures:

   ~lib.align.aligned_mask.BlurMask
   ~lib.align.aligned_mask.LandmarksMask
   ~lib.align.aligned_mask.Mask
   
.. rubric:: Module

.. automodule:: lib.align.aligned_mask
   :members:
   :undoc-members:
   :show-inheritance:


alignments module
=================

Handles alignments stored in a serialized alignments.fsa file

.. rubric:: Module Summary

.. autosummary::
   :nosignatures:

   ~lib.align.alignments.Alignments
   ~lib.align.alignments.Thumbnails

.. rubric:: Module

.. automodule:: lib.align.alignments
   :members:
   :undoc-members:
   :show-inheritance:


constants module
================
Holds various constants for use in generating and manipulating aligned face images

.. automodule:: lib.align.constants
   :members:
   :undoc-members:
   :show-inheritance:


detected\_face module
=====================

Handles detected face objects and their associated masks.

.. rubric:: Module Summary

.. autosummary::
   :nosignatures:

   ~lib.align.detected_face.DetectedFace
   ~lib.align.detected_face.update_legacy_png_header

.. rubric:: Module

.. automodule:: lib.align.detected_face
   :members:
   :undoc-members:
   :show-inheritance:


pose module
===========
Handles pose estimates based on aligned face data

.. automodule:: lib.align.pose
   :members:
   :undoc-members:
   :show-inheritance:


thumbnails module
=================
Handles creation of jpg thumbnails for storage in alignment files/png headers

.. automodule:: lib.align.thumbnails
   :members:
   :undoc-members:
   :show-inheritance:


updater module
==============
Handles the update of alignments files to the latest version

.. automodule:: lib.align.updater
   :members:
   :undoc-members:
   :show-inheritance:
