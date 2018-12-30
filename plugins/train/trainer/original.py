#!/usr/bin/env python3
""" Original Trainer """

from ._base import TrainerBase, logger


class Trainer(TrainerBase):
    """ Original Model Trainer """

    def process_training_opts(self):
        """ No specific training options for Original """
        pass
