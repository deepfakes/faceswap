#!/usr/bin/env python3
""" Masks functions for faceswap.py """

import inspect
import logging
import sys
import requests
import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_available_masks():
    """ Return a list of the available masks for cli """
    masks = ['components', 'dfl_full', 'facehull', 'none']
    logger.debug(masks)
    return masks


def get_default_mask():
    """ Set the default mask for cli """
    masks = get_available_masks()
    default = "dfl_full"
    default = default if default in masks else masks[0]
    logger.debug(default)
    return default


class Mask():
    """ Parent class for masks

        the output mask will be <mask_type>.mask
        channels: 1, 3 or 4:
                    1 - Returns a single channel mask
                    3 - Returns a 3 channel mask
                    4 - Returns the original image with the mask in the alpha channel """

    def __init__(self, landmarks, face, type, channels=4):
        logger.trace("Initializing %s: (face_shape: %s, channels: %s, landmarks: %s)",
                     self.__class__.__name__, face.shape, channels, landmarks)
        self.landmarks = landmarks
        self.face = face
        self.type = type
        self.channels = channels
        self.check_for_models(type)
        mask = self.build_mask()
        self.mask = self.merge_mask(mask)
        logger.trace("Initialized %s", self.__class__.__name__)

    def check_for_models(self, type):
        dirname = 'C:/data/models/'
        URL = 'https://docs.google.com/uc?export=download'
        filename = {'vgg_300':     'Nirkin_300_softmax',
                    'vgg_500':     'Nirkin_500_softmax',
                    'unet_256':    'DFL_256_sigmoid'}
        """
        URL = {'vgg_300':     https://drive.google.com/open?id=1_DxWEvcs8UwIBR-d7ga8-9WBlLOKDxa1 ,
               'vgg_500':     https://drive.google.com/open?id=1YY1-l4L37VwsWx1MHIaamG0xmnYHQpiT ,
               'unet_256':    https://drive.google.com/open?id=1LSn-jf9O6VjeYexfNdd4hOG6AtbpDA3O }
        """
        id = {'vgg_300':     '1_DxWEvcs8UwIBR-d7ga8-9WBlLOKDxa1',
              'vgg_500':     '1YY1-l4L37VwsWx1MHIaamG0xmnYHQpiT',
              'unet_256':    '1LSn-jf9O6VjeYexfNdd4hOG6AtbpDA3O'}
        expected_location = Path(dirname, filename[type]).with_suffix('.h5')
        if not expected_location.is_file():
            download_model(id[type], expected_location, URL)
        else:



    def merge_mask(self, mask):
        """ Return the mask in requested shape """
        logger.trace("mask_shape: %s", mask.shape)
        assert self.channels in (1, 3, 4), "Channels should be 1, 3 or 4"
        assert mask.shape[2] == 1 and mask.ndim == 3, "Input mask be 3 dimensions with 1 channel"

        if self.channels == 3:
            retval = np.tile(mask, 3)
        elif self.channels == 4:
            retval = np.concatenate((self.face, mask), -1)
        else:
            retval = mask

        logger.trace("Final mask shape: %s", retval.shape)
        return retval

    def build_mask(self):
        """ Build the mask """
        mask = np.zeros(self.face.shape[0:-1] + (1, ), dtype='float32')
        mask_dict = {'facehull':    self.one_part_facehull,
                     'dfl_full':    self.three_part_facehull,
                     'components':  self.eight_part_facehull,
                     'none':        self.default,
                     'vgg_300':     self.nirkin_300,
                     'vgg_500':     self.nirkin_500,
                     'unet_256':    self.ternaus_256}
        mask_function, use_facehull = mask_dict[self.type]
        mask = mask_function(mask)
        return mask

    def one_part_facehull(self, mask):
        """ Basic facehull mask """
        #hull = cv2.convexHull(np.array(self.landmarks).reshape((-1, 2)))
        parts = [(self.landmarks)]
        mask = self.compute_facehull(mask, parts)
        return mask

    def three_part_facehull(self, mask):
        """ DFL facehull mask """
        nose_ridge = (self.landmarks[27:31], self.landmarks[33:34])
        jaw = (self.landmarks[0:17],
               self.landmarks[48:68],
               self.landmarks[0:1],
               self.landmarks[8:9],
               self.landmarks[16:17])
        eyes = (self.landmarks[17:27],
                self.landmarks[0:1],
                self.landmarks[27:28],
                self.landmarks[16:17],
                self.landmarks[33:34])
        parts = [jaw, nose_ridge, eyes]
        mask = self.compute_facehull(mask, parts)
        return mask

    def eight_part_facehull(self, mask):
        """ Component facehull mask """
        r_jaw = (self.landmarks[0:9], self.landmarks[17:18])
        l_jaw = (self.landmarks[8:17], self.landmarks[26:27])
        r_cheek = (self.landmarks[17:20], self.landmarks[8:9])
        l_cheek = (self.landmarks[24:27], self.landmarks[8:9])
        nose_ridge = (self.landmarks[19:25], self.landmarks[8:9],)
        r_eye = (self.landmarks[17:22],
                 self.landmarks[27:28],
                 self.landmarks[31:36],
                 self.landmarks[8:9])
        l_eye = (self.landmarks[22:27],
                 self.landmarks[27:28],
                 self.landmarks[31:36],
                 self.landmarks[8:9])
        nose = (self.landmarks[27:31], self.landmarks[31:36])
        parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]
        mask = self.compute_facehull(mask, parts)
        return mask

    @staticmethod
    def compute_facehull(self, mask, parts):
        """ Compute the facehull """
        for item in parts:
            hull = cv2.convexHull(np.concatenate(item))  # pylint: disable=no-member
            cv2.fillConvexPoly(mask, hull, 1., lineType=cv2.LINE_AA)  # pylint: disable=no-member
        return mask

    def default(self, mask):
        """ Basic facehull mask """
        mask = 1.
        print(default.shape)
        return mask

    def nirkin_500(self, mask):
        """ Basic facehull mask """
        mask = 1.
        return mask

    def nirkin_300(self, mask):
        """ Basic facehull mask """
        mask = 1.
        return mask

    def ternaus_256(self, mask):
        """ Basic facehull mask """
        mask = 1.
        return mask

    @staticmethod
    def download_model(self, id, destination, URL):
        CHUNK_SIZE = 32768
        session = requests.Session()
        response = session.get(URL, params = { 'id' : id }, stream = True)
        for key, value in response.cookies.items():
            token = value if key.startswith('download_warning') else None
        if token:
            params = { 'id' : id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

