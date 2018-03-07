import cv2
import numpy
from random import shuffle

from .utils import BackgroundGenerator
from .umeyama import umeyama

class TrainingDataGenerator():
    def __init__(self, random_transform_args, coverage, scale=5, zoom=1): #TODO thos default should stay in the warp function
        self.random_transform_args = random_transform_args
        self.coverage = coverage
        self.scale = scale
        self.zoom = zoom

    def minibatchAB(self, images, batchsize):
        batch = BackgroundGenerator(self.minibatch(images, batchsize), 1)
        for ep1, warped_img, target_img in batch.iterator():
            yield ep1, warped_img, target_img

    # A generator function that yields epoch, batchsize of warped_img and batchsize of target_img
    def minibatch(self, data, batchsize):
        length = len(data)
        assert length >= batchsize, "Number of images is lower than batch-size (Note that too few images may lead to bad training). # images: {}, batch-size: {}".format(length, batchsize)
        epoch = i = 0
        shuffle(data)
        while True:
            size = batchsize
            if i+size > length:
                shuffle(data)
                i = 0
                epoch+=1
            rtn = numpy.float32([self.read_image(img) for img in data[i:i+size]])
            i+=size
            yield epoch, rtn[:,0,:,:,:], rtn[:,1,:,:,:]       

    def color_adjust(self, img):
        return img / 255.0
    
    def read_image(self, fn):
        try:
            image = self.color_adjust(cv2.imread(fn))
        except TypeError:
            raise Exception("Error while reading image", fn)
        
        image = cv2.resize(image, (256,256))
        image = self.random_transform( image, **self.random_transform_args )
        warped_img, target_img = self.random_warp( image, self.coverage, self.scale, self.zoom )
        
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
    def random_warp(self, image, coverage, scale = 5, zoom = 1):
        assert image.shape == (256, 256, 3)
        range_ = numpy.linspace(128 - coverage//2, 128 + coverage//2, 5)
        mapx = numpy.broadcast_to(range_, (5, 5))
        mapy = mapx.T

        mapx = mapx + numpy.random.normal(size=(5,5), scale=scale)
        mapy = mapy + numpy.random.normal(size=(5,5), scale=scale)

        interp_mapx = cv2.resize(mapx, (80*zoom,80*zoom))[8*zoom:72*zoom,8*zoom:72*zoom].astype('float32')
        interp_mapy = cv2.resize(mapy, (80*zoom,80*zoom))[8*zoom:72*zoom,8*zoom:72*zoom].astype('float32')

        warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

        src_points = numpy.stack([mapx.ravel(), mapy.ravel() ], axis=-1)
        dst_points = numpy.mgrid[0:65*zoom:16*zoom,0:65*zoom:16*zoom].T.reshape(-1,2)
        mat = umeyama(src_points, dst_points, True)[0:2]

        target_image = cv2.warpAffine(image, mat, (64*zoom,64*zoom))

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
