#!/usr/bin/env python3
""" Package for handling alignments files, detected faces and aligned faces along with their
associated objects. """
from __future__ import annotations
import typing as T

from .augmentation import ImageAugmentation
from .generator import Feeder
from .lr_finder import LearningRateFinder
from .preview_cv import PreviewBuffer, TriggerType

if T.TYPE_CHECKING:
    from .preview_cv import PreviewBase
    Preview: type[PreviewBase]

try:
    from .preview_tk import PreviewTk as Preview
except ImportError:
    from .preview_cv import PreviewCV as Preview
