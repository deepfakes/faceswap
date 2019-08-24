#!/usr/bin/env python3
""" xxxxxxx for faceswap.py
Adapted from: xxxxxxxxxxxx
MIT License

Copyright (c) xxxx xxx xxxxxxx

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np

from ._base import Masker, logger


class Mask(Masker):
    """ Perform transformation to align and get landmarks """
    def __init__(self, **kwargs):
        git_model_id = None
        model_filename = None
        super().__init__(git_model_id=git_model_id,
                         model_filename=model_filename,
                         input_size=256,
                         **kwargs)
        self.vram = 0
        self.model = None
        self.supports_plaidml = True

    def initialize(self, *args, **kwargs):
        """ Initialization tasks to run prior to alignments """
        try:
            super().initialize(*args, **kwargs)
            logger.info("Initializing Dummy Mask Model...")
            logger.debug("Dummy initialize: (args: %s kwargs: %s)", args, kwargs)
            self.init.set()
            logger.info("Initialized Dummy Mask Model")
        except Exception as err:
            self.error.set()
            raise err

    # MASK PROCESSING
    def build_masks(self, faces, landmarks):
        """ Function for creating facehull masks
            Faces may be of shape (batch_size, height, width, 3) or (height, width, 3)
        """
        masks = np.full(faces.shape[:-1] + (1,), fill_value=255, dtype='uint8')
        faces = np.concatenate((faces[..., :3], masks), axis=-1)
        return faces, masks



    
