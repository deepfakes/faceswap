**************
manual package
**************

.. contents:: Contents
   :depth: 1
   :local:

Subpackages
===========
The following subpackages handle the main two display areas of the Manual Tool's GUI.

.. toctree::
   :maxdepth: 2

   manual.faceviewer
   manual.frameviewer

manual module
=============
The Manual Module is the main entry point into the Manual Editor Tool.

Module Summary
--------------
.. automodsumm:: tools.manual.manual
   :classes-only:
   :skip: ControlPanel, DetectedFaces, DisplayFrame, ExtractMedia, Extractor, FacesFrame, FrameNavigation, MultiThread

Module
------
.. automodule:: tools.manual.manual
   :members:
   :undoc-members:
   :show-inheritance:

detected_faces module
=====================
Module Summary
--------------
.. automodsumm:: tools.manual.detected_faces
   :classes-only:
   :skip: AlignerExtract, Alignments, DetectedFace, deepcopy, logger

Module
------
.. automodule:: tools.manual.detected_faces
   :members:
   :undoc-members:
   :show-inheritance:
