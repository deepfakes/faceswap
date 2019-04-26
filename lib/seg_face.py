""" doc string """
import os
import sys
import math
import inspect
import pathlib
import cv2
import numpy as np
import keras
from keras.layers import *
from keras.models import Model

CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

class FaceSegmentation():
    """ doc string """

    def __init__(self, batch_size=24, model_img_size=(300, 300),
                 model_path='C:/data/models/Nirkin_500.h5',
                 in_dir='C:/data/putin/'):
        """ doc string """
        self.model = keras.models.load_model(model_path)
        #keras.models.save_model(self.model, 'C:/data/models/compressed_segmentation_model.h5', include_optimizer=False)
        print('\n' + 'Model loaded')

        in_dir = pathlib.Path(in_dir)
        image_file_list = self.get_image_paths(in_dir)
        self.num_images = len(image_file_list)
        image_dataset, self.means = self.dataset_setup(image_file_list, model_img_size, batch_size)
        self.memmapped_images = np.load(image_dataset, mmap_mode='r+')
        print('\n' + 'Image dataset loaded')
        """
        import requests

        def download_file_from_google_drive(id, destination):
            URL = "https://docs.google.com/uc?export=download"

            session = requests.Session()

            response = session.get(URL, params = { 'id' : id }, stream = True)
            token = get_confirm_token(response)

            if token:
                params = { 'id' : id, 'confirm' : token }
                response = session.get(URL, params = params, stream = True)

            save_response_content(response, destination)    

        def get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value

            return None

        def save_response_content(response, destination):
            CHUNK_SIZE = 32768

            with open(destination, "wb") as f:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
        In [2]:
        file_id = '0B1fGSuBXAh1IeEpzajRISkNHckU'
        destination = '/home/myusername/work/myfile.ext'
        download_file_from_google_drive(file_id, destination)
        """

    def segment(self, model_type, out_dir='C:/data/masked/'):
        """ doc string """
        i = 0
        for img_batch, avgs in zip(self.memmapped_images, self.means):
            if model_type=='Nirkin':
                results = self.model.predict_on_batch(img_batch)
                img_batch += avgs[:, None, None, :]
                mask_batch = np.clip(results.argmax(axis=-1), 0, 1).astype('uint8')
                mask_batch = np.expand_dims(mask_batch, axis=-1)
                generator = (self.postprocessing(mask[:, :, 0:1]) for mask in mask_batch)
                mask_batch = np.array(tuple(generator))
                img_batch = np.concatenate((img_batch.astype('uint8'), mask_batch), axis=-1)
            if  model_type=='DFL':
                img_batch += avgs[:, None, None, :]
                results = self.model.predict(img_batch / 255.)
                results[results < 0.1] = 0.
                mask_batch = results * 255.
                img_batch = np.concatenate((img_batch, mask_batch), axis=-1).astype('uint8')

            for four_channel in img_batch:
                if i < self.num_images:
                    path_string = '{0}{1:05d}.png'.format(out_dir, i)
                    cv2.imwrite(path_string, four_channel)  # pylint: disable=no-member
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
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # pylint: disable=no-member
            cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)  # pylint: disable=no-member
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)  # pylint: disable=no-member
            
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)  # pylint: disable=no-member
            cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)  # pylint: disable=no-member

        mask *= 255
        #Fill holes
        fill_holes = True
        if fill_holes:
            not_holes = mask.copy()
            not_holes = np.pad(not_holes, ((2, 2), (2, 2), (0, 0)), 'constant')
            cv2.floodFill(not_holes, None, (0, 0), 255)  # pylint: disable=no-member
            holes = cv2.bitwise_not(not_holes)[2:-2,2:-2]  # pylint: disable=no-member
            mask = cv2.bitwise_or(mask, holes)  # pylint: disable=no-member
            mask = np.expand_dims(mask, axis=-1)
            
        return mask

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


    def merge_comparision(self, out_dir='C:/data/masked/'):
        """ doc string """
        image_file_lists = []
        target_size = 300
        for in_dir in ['C:/data/masked - DFL/','C:/data/masked - Nirkin500/','C:/data/masked - Nirkin300/']:
            dir = pathlib.Path(in_dir)
            image_file_lists.append(self.get_image_paths(dir))
        
        for i, (file_a, file_b, file_c) in enumerate(zip(image_file_lists[0], image_file_lists[1], image_file_lists[2])):
            img_a = cv2.imread(file_a, cv2.IMREAD_UNCHANGED).astype('float32')  # pylint: disable=no-member
            height, width, _ = img_a.shape
            image_size = min(height, width)
            method = cv2.INTER_CUBIC if image_size < target_size else cv2.INTER_AREA  # pylint: disable=no-member
            img_a = cv2.resize(img_a, (target_size, target_size), method)  # pylint: disable=no-member
            
            img_b = cv2.imread(file_b, cv2.IMREAD_UNCHANGED).astype('float32')  # pylint: disable=no-member
            height, width, _ = img_b.shape
            image_size = min(height, width)
            method = cv2.INTER_CUBIC if image_size < target_size else cv2.INTER_AREA  # pylint: disable=no-member
            img_b = cv2.resize(img_b, (target_size, target_size), method)  # pylint: disable=no-member
            
            img_c = cv2.imread(file_c, cv2.IMREAD_UNCHANGED).astype('float32')  # pylint: disable=no-member
            height, width, _ = img_c.shape
            image_size = min(height, width)
            method = cv2.INTER_CUBIC if image_size < target_size else cv2.INTER_AREA  # pylint: disable=no-member
            img_c = cv2.resize(img_c, (target_size, target_size), method)  # pylint: disable=no-member
            
            compare = np.concatenate((img_a, img_b, img_c), axis=1)
            path_string = '{0}{1:05d}.png'.format(out_dir, i)
            cv2.imwrite(path_string, compare)  # pylint: disable=no-member

if __name__ == '__main__':
    Segmentator = FaceSegmentation(model_path='C:/data/models/Nirkin_300.h5',
                                   model_img_size=(300, 300),
                                   batch_size=8,
                                   in_dir='C:/data/putin/')
    #Segmentator.segment(out_dir='C:/data/masked/', model_type='Nirkin')
    Segmentator.merge_comparision(out_dir='C:/data/masked/')
