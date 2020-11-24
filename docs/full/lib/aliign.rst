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
   ~lib.align.aligned_face.PoseEstimate
   ~lib.align.aligned_face.transform_image

.. rubric:: Module

.. automodule:: lib.align.aligned_face
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

detected\_face module
=====================

Handles detected face objects and their associated masks.

.. rubric:: Module Summary

.. autosummary::
   :nosignatures:
   
   ~lib.align.detected_face.BlurMask
   ~lib.align.detected_face.DetectedFace
   ~lib.align.detected_face.Mask

.. rubric:: Module

.. automodule:: lib.align.detected_face
   :members:
   :undoc-members:
   :show-inheritance:
