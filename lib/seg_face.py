""" doc string """
import os
import sys
import math
import inspect
import pathlib
import cv2
import numpy as np
import keras
CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

class FaceSegmentation():
    """ doc string """

    def __init__(self, batch_size=24, model_img_size=(300, 300),
                 model_path='C:/data/models/segmentation_model.h5',
                 in_dir='C:/data/putin/'):
        """ doc string """
        self.model = keras.models.load_model(model_path)
        print('\n' + 'Model loaded')

        in_dir = pathlib.Path(in_dir)
        image_file_list = self.get_image_paths(in_dir)
        self.num_images = len(image_file_list)
        image_dataset, self.means = self.dataset_setup(image_file_list, model_img_size, batch_size)
        self.memmapped_images = np.load(image_dataset, mmap_mode='r+')
        print('\n' + 'Image dataset loaded')

    def segment(self, out_dir='C:/data/masked/'):
        """ doc string """
        i = 0
        for img_batch, avgs in zip(self.memmapped_images, self.means):
            results = self.model.predict_on_batch(img_batch)
            img_batch += avgs[:, None, None, :]
            mask_batch = results.argmax(axis=-1).astype('float32')
            mask_batch = np.expand_dims(mask_batch, axis=-1)
            generator = (self.postprocessing(mask[:, :, 0:1]) for mask in mask_batch)
            mask_batch = np.array(tuple(generator))
            img_batch = np.concatenate((img_batch, mask_batch), axis=-1)

            for four_channel in img_batch:
                if i < self.num_images:
                    path_string = '{0}{1:05d}.png'.format(out_dir, i)
                    cv2.imwrite(path_string, four_channel.astype('uint8'))  # pylint: disable=no-member
                    i += 1

        print('\n' + 'Mask generation complete')

    @staticmethod
    def get_image_paths(directory):
        """ doc string """
        image_extensions = [".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"]
        dir_scanned = sorted(os.scandir(directory), key=lambda x: x.name)
        dir_contents = []

        for chkfile in dir_scanned:
            if any([chkfile.name.lower().endswith(ext) for ext in image_extensions]):
                dir_contents.append(chkfile.path)
        return dir_contents

    def dataset_setup(self, img_list, in_size, batch_size):
        """ doc string """
        filename = str(pathlib.Path(img_list[0]).parents[0].joinpath(('Images_Batched.npy')))
        batch_num = int(math.ceil(len(img_list) / batch_size))
        img_shape = (batch_num, batch_size, in_size[0], in_size[1], 3)
        means = np.empty((batch_num, batch_size, 3), dtype='float32')

        dataset = np.lib.format.open_memmap(filename, mode='w+', dtype='float32', shape=img_shape)
        for i, (img, avg) in enumerate(self.loader(img_file, in_size[0]) for img_file in img_list):
            dataset[i // batch_size, i % batch_size] = img
            means[i // batch_size, i % batch_size] = avg
        del dataset
        return filename, means

    @staticmethod
    def loader(img_file, target_size):
        """ doc string """
        img = cv2.imread(img_file).astype('float32')  # pylint: disable=no-member
        height, width, _ = img.shape
        image_size = min(height, width)
        method = cv2.INTER_CUBIC if image_size < target_size else cv2.INTER_AREA  # pylint: disable=no-member
        img = cv2.resize(img, (target_size, target_size), method)  # pylint: disable=no-member
        avg = np.mean(img, axis=(0, 1))
        img -= avg
        return img, avg

    @staticmethod
    def postprocessing(mask):
        """ doc string """
        #Select_largest_segment
        pop_small_segments = False # Don't do this right now
        if pop_small_segments:
            results = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)  # pylint: disable=no-member
            _, labels, stats, _ = results
            segments_ranked_by_area = np.argsort(stats[:, -1])[::-1]
            mask[labels != segments_ranked_by_area[0, 0]] = 0.

        #Smooth contours
        smooth_contours = False # Don't do this right now
        if smooth_contours:
            iters = 2
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # pylint: disable=no-member
            for _ in range(iters):
                cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)  # pylint: disable=no-member
                cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)  # pylint: disable=no-member

        #Fill holes
        fill_holes = False # Don't do this right now
        if fill_holes:
            inversion = mask.copy()
            inversion = np.pad(inversion, ((2, 2), (2, 2), (0, 0)), 'constant')
            cv2.floodFill(inversion, None, (0, 0), 1)  # pylint: disable=no-member
            holes = cv2.bitwise_not(inversion)[2:-2, 2:-2]  # pylint: disable=no-member
            mask = cv2.bitwise_or(mask, holes)  # pylint: disable=no-member
            mask = np.expand_dims(mask, axis=-1)
        return mask * 255

    @staticmethod
    def crop_standarization(image, landmarks):
        """ doc string """
        box_xy, width_height, _ = cv2.boundingRect(landmarks)  # pylint: disable=no-member
        longest = max(width_height) * 1.2
        moments = cv2.moments(landmarks)  # pylint: disable=no-member
        width = longest + abs(box_xy[0] - moments["m10"] // moments["m00"])
        height = longest + abs(box_xy[1] - moments["m01"] // moments["m00"])

        xmin = int(max(0, box_xy[0] - width / 2.))
        ymin = int(max(0, box_xy[1] - height / 2.))
        xmax = int(min(image.shape[1], box_xy[0] + width / 2.))
        ymax = int(min(image.shape[0], box_xy[1] + height / 2.))
        rect = slice(ymin, ymax), slice(xmin, xmax)
        return rect


if __name__ == '__main__':
    Segmentator = FaceSegmentation(model_path='C:/data/models/segmentation_model.h5',
                                   in_dir='C:/data/putin/')
    Segmentator.segment(out_dir='C:/data/masked/')
