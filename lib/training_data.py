import cv2
import numpy
from random import shuffle

from .umeyama import umeyama

random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4,
}

# GAN
# random_transform_args = {
#     'rotation_range': 20,
#     'zoom_range': 0.1,
#     'shift_range': 0.05,
#     'random_flip': 0.5,
#     }
def read_image(fn, random_transform_args=random_transform_args):
    image = cv2.imread(fn)
    image = cv2.resize(image, (256,256)) / 255 * 2 - 1
    image = random_transform( image, **random_transform_args )
    warped_img, target_img = random_warp( image )
    
    return warped_img, target_img

# A generator function that yields epoch, batchsize of warped_img and batchsize of target_img
def minibatch(data, batchsize):
    length = len(data)
    epoch = i = 0
    shuffle(data)
    while True:
        size = batchsize
        if i+size > length:
            shuffle(data)
            i = 0
            epoch+=1        
        rtn = numpy.float32([read_image(data[j]) for j in range(i,i+size)])
        i+=size
        yield epoch, rtn[:,0,:,:,:], rtn[:,1,:,:,:]       

def minibatchAB(images, batchsize):
    batch = BackgroundGenerator(minibatch(images, batchsize))
    for ep1, warped_img, target_img in batch.iterator():
        yield ep1, warped_img, target_img

def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):
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
def random_warp(image):
    assert image.shape == (256, 256, 3)
    range_ = numpy.linspace(128 - 80, 128 + 80, 5)
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

def get_transpose_axes(n):
    if n % 2 == 0:
        y_axes = list(range(1, n - 1, 2))
        x_axes = list(range(0, n - 1, 2))
    else:
        y_axes = list(range(0, n - 1, 2))
        x_axes = list(range(1, n - 1, 2))
    return y_axes, x_axes, [n - 1]

def stack_images(images):
    images_shape = numpy.array(images.shape)
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [numpy.prod(images_shape[x]) for x in new_axes]
    return numpy.transpose(
        images,
        axes=numpy.concatenate(new_axes)
        ).reshape(new_shape)

# From: https://stackoverflow.com/questions/7323664/python-generator-pre-fetch
import threading
import queue as Queue
class BackgroundGenerator(threading.Thread):
    def __init__(self, generator):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(1)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        # Put until queue size is reached. Note: put blocks only if put is called while queue has already reached max size
        # => this makes 2 prefetched items! One in the queue, one waiting for insertion!
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def iterator(self):
        while True:
            next_item = self.queue.get()
            if next_item is None:
                break
            yield next_item