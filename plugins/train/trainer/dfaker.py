#!/usr/bin/env python3
""" DFaker Trainer """

from ._base import TrainerBase, logger


class Trainer(TrainerBase):
    """ Dfaker Trainer """

    def process_transform_kwargs(self):
        """ Override for specific image manipulation kwargs
            See lib.training_data.ImageManipulation() for valid kwargs"""
        warped_zoom = self.model.input_shape[0] // 64
        target_zoom = self.model.predictors["a"].output_shape[0][1] // 64
        transform_kwargs = {"rotation_range": 10,
                            "zoom_range": 0.05,
                            "shift_range": 0.05,
                            "random_flip": 0.5,
                            "zoom": (warped_zoom, target_zoom),
                            "coverage": 160,
                            "scale": 5}
        logger.debug(transform_kwargs)
        return transform_kwargs
