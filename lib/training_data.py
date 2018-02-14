import cv2
import numpy
from random import shuffle

from .utils import BackgroundGenerator
from .umeyama import umeyama

class TrainingDataGenerator():
    def __init__(self, random_transform_args, coverage):
        self.random_transform_args = random_transform_args
        self.coverage = coverage

    def load_images(self, image_paths):
        iter_all_images = (cv2.imread(fn) for fn in image_paths)
        for i, image in enumerate(iter_all_images):
            if i == 0:
                all_images = numpy.empty((len(image_paths),) + image.shape, dtype=image.dtype)
            all_images[i] = image

        try:
            first = all_images[i]
        except:
            print('Cannot find images. Make sure the data directory paths are properly entered.')

        return all_images

    def minibatchAB(self, images_A, images_B, batchsize):
        images_A = self.load_images(images_A) / 255.0
        images_B = self.load_images(images_B) / 255.0
        images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2)) # color matching
        
        batch = BackgroundGenerator(self.minibatch(images_A, images_B, batchsize), 1)
        for ep1, warped_img_A, target_img_A, warped_img_B, target_img_B in batch.iterator():
            yield ep1, warped_img_A, target_img_A, warped_img_B, target_img_B

    # A generator function that yields epoch, batchsize of warped_img and batchsize of target_img
    def minibatch(self, images_A, images_B, batchsize):
        length_A = len(images_A)
        length_B = len(images_B)
        assert length_A >= batchsize, "Number of images a is lower than batch-size (Note that too few images may lead to bad training). # images: {}, batch-size: {}".format(length_A, batchsize)
        assert length_B >= batchsize, "Number of images b is lower than batch-size (Note that too few images may lead to bad training). # images: {}, batch-size: {}".format(length_B, batchsize)
        
        epoch = index_A = index_B = 0
        shuffle(images_A)
        
        while True:
            overflow=False
            
            if index_A+batchsize > length_A:
                shuffle(images_A)
                index_A = 0
                overflow=True
            
            if index_B+batchsize > length_B:
                shuffle(images_B)
                index_B = 0
                overflow=True
                
            if overflow:
                epoch+=1
                
            rtn_A = numpy.float32([self.warp_image(img) for img in images_A[index_A:index_A+batchsize]])
            index_A+=batchsize
            
            rtn_B = numpy.float32([self.warp_image(img) for img in images_B[index_B:index_B+batchsize]])
            index_B+=batchsize            
            
            yield epoch, rtn_A[:,0,:,:,:], rtn_A[:,1,:,:,:], rtn_B[:,0,:,:,:], rtn_B[:,1,:,:,:]
    
    def warp_image(self, image):
        image = self.random_transform( image, **self.random_transform_args )
        warped_img, target_img = self.random_warp( image, self.coverage )
        
        return warped_img, target_img

    def random_transform(self, image, rotation_range, zoom_range, shift_range, random_flip):
        h, w = image.shape[0:2]
        rotation = numpy.random.uniform(-rotation_range, rotation_range)
        scale = numpy.random.uniform(1 - zoom_range, 1 + zoom_range)
        tx = numpy.random.uniform(-shift_range, shift_range) * w
        ty = numpy.random.uniform(-shift_range, shift_range) * h
        mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
        mat[:, 2] += (tx, ty)
        result = cv2.warpAffine(
            image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if numpy.random.random() < random_flip:
            result = result[:, ::-1]
        return result

    # get pair of random warped images from aligned face image
    def random_warp(self, image, coverage):
        assert image.shape == (256, 256, 3)
        range_ = numpy.linspace(128 - coverage//2, 128 + coverage//2, 5)
        mapx = numpy.broadcast_to(range_, (5, 5))
        mapy = mapx.T

        mapx = mapx + numpy.random.normal(size=(5, 5), scale=5)
        mapy = mapy + numpy.random.normal(size=(5, 5), scale=5)

        interp_mapx = cv2.resize(mapx, (80, 80))[8:72, 8:72].astype('float32')
        interp_mapy = cv2.resize(mapy, (80, 80))[8:72, 8:72].astype('float32')

        warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

        src_points = numpy.stack([mapx.ravel(), mapy.ravel()], axis=-1)
        dst_points = numpy.mgrid[0:65:16, 0:65:16].T.reshape(-1, 2)
        mat = umeyama(src_points, dst_points, True)[0:2]

        target_image = cv2.warpAffine(image, mat, (64, 64))

        return warped_image, target_image

def stack_images(images):
    def get_transpose_axes(n):
        if n % 2 == 0:
            y_axes = list(range(1, n - 1, 2))
            x_axes = list(range(0, n - 1, 2))
        else:
            y_axes = list(range(0, n - 1, 2))
            x_axes = list(range(1, n - 1, 2))
        return y_axes, x_axes, [n - 1]
    
    images_shape = numpy.array(images.shape)
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [numpy.prod(images_shape[x]) for x in new_axes]
    return numpy.transpose(
        images,
        axes=numpy.concatenate(new_axes)
        ).reshape(new_shape)
