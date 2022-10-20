#!/usr/bin/env python3
""" Package for handling alignments files, detected faces and aligned faces along with their
associated objects. """

from typing import Type, TYPE_CHECKING

from .augmentation import ImageAugmentation
from .generator import PreviewDataGenerator, TrainingDataGenerator
from .preview_cv import PreviewBuffer, TriggerType

if TYPE_CHECKING:
    from .preview_cv import PreviewBase
    Preview: Type[PreviewBase]

try:
    from .preview_tk import PreviewTk as Preview
except ImportError:
    from .preview_cv import PreviewCV as Preview
