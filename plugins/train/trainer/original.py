#!/usr/bin/env python3
""" Original Trainer """

from lib.utils import get_module_objects
from ._base import TrainerBase


class Trainer(TrainerBase):
    """ Original is currently identical to Base """
    def __init__(self, *args, **kwargs):  # pylint:disable=useless-super-delegation
        super().__init__(*args, **kwargs)


__all__ = get_module_objects(__name__)
