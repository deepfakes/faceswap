import cv2
import numpy
import os

def get_image_paths( directory ):
    return [ x.path for x in os.scandir( directory ) if x.name.endswith(".jpg") or x.name.endswith(".png") ]

def load_images( image_paths, convert=None ):
    iter_all_images = ( cv2.imread(fn) for fn in image_paths )
    if convert:
        iter_all_images = ( convert(img) for img in iter_all_images )
    for i,image in enumerate( iter_all_images ):
        if i == 0:
            all_images = numpy.empty( ( len(image_paths), ) + image.shape, dtype=image.dtype )
        all_images[i] = image
    return all_images

def get_transpose_axes( n ):
    if n % 2 == 0:
        y_axes = list( range( 1, n-1, 2 ) )
        x_axes = list( range( 0, n-1, 2 ) )
    else:
        y_axes = list( range( 0, n-1, 2 ) )
        x_axes = list( range( 1, n-1, 2 ) )
    return y_axes, x_axes, [n-1]

def stack_images( images ):
    images_shape = numpy.array( images.shape )
    new_axes = get_transpose_axes( len( images_shape ) )
    new_shape = [ numpy.prod( images_shape[x] ) for x in new_axes ]
    return numpy.transpose(
        images,
        axes = numpy.concatenate( new_axes )
        ).reshape( new_shape )
