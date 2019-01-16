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
        transform_kwargs = {"zoom": (warped_zoom, target_zoom),
                            "coverage_ratio": 0.625}
        logger.debug(transform_kwargs)
        return transform_kwargs
