import cv2
import time
import numpy
import random

from lib.utils import BackgroundGenerator
from lib.training_data import TrainingDataGenerator, stack_images
from .image_augmentation import random_transform, random_warp_src_dest
from .utils import load_images_aligned

class DFTrainingDataGenerator(TrainingDataGenerator):
    def __init__(self, random_transform_args, coverage):
        super().__init__(random_transform_args, coverage)

    def minibatchAB(self, images, srcPoints, dstPoints, batch_size ):
        batch = BackgroundGenerator(self.minibatch(images, srcPoints, dstPoints, batch_size), 1)
        return batch.iterator()

    def minibatch(self, images, srcPoints, dstPoints, batch_size):
        epoch = 0
        while True:
            epoch+=1
            indices = numpy.random.choice(range(0,images.shape[0]),size=batch_size,replace=False)
            for i,index in enumerate(indices):
                image = images[index]
                image = random_transform( image, **self.random_transform_args )

                closest = ( numpy.mean(numpy.square(srcPoints[index]-dstPoints),axis=(1,2)) ).argsort()[:10]
                closest = numpy.random.choice(closest)
                warped_img, target_img, mask_image = random_warp_src_dest( image,srcPoints[index],dstPoints[ closest ] )
        
                if numpy.random.random() < 0.5:
                    warped_img = warped_img[:,::-1]
                    target_img = target_img[:,::-1]
                    mask_image = mask_image[:,::-1]

                if i == 0:
                    warped_images = numpy.empty( (batch_size,) + warped_img.shape, warped_img.dtype )
                    target_images = numpy.empty( (batch_size,) + target_img.shape, warped_img.dtype )
                    mask_images = numpy.empty( (batch_size,)   + mask_image.shape, mask_image.dtype )

                warped_images[i] = warped_img
                target_images[i] = target_img
                mask_images[i]   = mask_image

            yield epoch, warped_images, target_images, mask_images

class Trainer():
    random_transform_args = {
        'rotation_range': 10,
        'zoom_range': 0.05,
        'shift_range': 0.05,
        'random_flip': 0.0,
        }

    def __init__(self, model, fn_A, fn_B, batch_size, *args):
        self.batch_size = batch_size
        self.model = model

        generator = DFTrainingDataGenerator(self.random_transform_args, 160)
        
        minImages = 2000#min(len(fn_A),len(fn_B))*20

        random.shuffle(fn_A)
        random.shuffle(fn_B)

        #NOTE this loads all images so it may be memory intensive! (cumber of images is maxed to 'minImages')
        images_A, landmarks_A = load_images_aligned(fn_A[:minImages])
        images_B, landmarks_B = load_images_aligned(fn_B[:minImages])

        images_A = images_A/255.0
        images_B = images_B/255.0

        images_A[:,:,:3] += images_B[:,:,:3].mean( axis=(0,1,2) ) - images_A[:,:,:3].mean( axis=(0,1,2) )

        self.images_A = generator.minibatchAB(images_A, landmarks_A, landmarks_B, self.batch_size)
        self.images_B = generator.minibatchAB(images_B, landmarks_B, landmarks_A, self.batch_size)

    def train_one_step(self, iter, viewer):
        epoch, warped_A, target_A, mask_A = next(self.images_A)
        epoch, warped_B, target_B, mask_B = next(self.images_B)
      
        #omask = numpy.ones((target_A.shape[0],64,64,1),float)

        loss_A = self.model.autoencoder_A.train_on_batch([warped_A,mask_A], [target_A,mask_A])
        loss_B = self.model.autoencoder_B.train_on_batch([warped_B,mask_B], [target_B,mask_B])

        print("Loss A [{}] Loss B [{}]".format(loss_A,loss_B),
            end='\r')

        if viewer is not None:
            viewer(self.show_sample(target_A[0:8,:,:,:3], target_B[0:8,:,:,:3]), "training")
            viewer(self.show_warped(warped_A[:6],warped_B[:6]), "warped")

    def show_warped(self, warped_A, warped_B):
        figWarped = numpy.stack([warped_A, warped_B],axis=0 )
        figWarped = numpy.clip( figWarped * 255, 0, 255 ).astype('uint8')
        figWarped = stack_images( figWarped )
        return figWarped

    def show_sample(self, test_A, test_B):
        test_A_i = []
        test_B_i = []
        
        for i in test_A:
            test_A_i.append(cv2.resize(i,(64,64),cv2.INTER_AREA))
        test_A_i = numpy.array(test_A_i).reshape((-1,64,64,3))

        for i in test_B:
            test_B_i.append(cv2.resize(i,(64,64),cv2.INTER_AREA))
        test_B_i = numpy.array(test_B_i).reshape((-1,64,64,3))
        
        zmask = numpy.zeros((test_A.shape[0],128,128,1),float)

        pred_a_a,pred_a_a_m = self.model.autoencoder_A.predict([test_A_i,zmask])
        pred_b_a,pred_b_a_m = self.model.autoencoder_B.predict([test_A_i,zmask])

        pred_a_b,pred_a_b_m = self.model.autoencoder_A.predict([test_B_i,zmask])
        pred_b_b,pred_b_b_m = self.model.autoencoder_B.predict([test_B_i,zmask])

        pred_a_a = pred_a_a[0:18,:,:,:3]
        pred_a_b = pred_a_b[0:18,:,:,:3]
        pred_b_a = pred_b_a[0:18,:,:,:3]
        pred_b_b = pred_b_b[0:18,:,:,:3]

        figure_A = numpy.stack([
            test_A,
            pred_a_a,
            pred_b_a,
            ], axis=1 )
        figure_B = numpy.stack([
            test_B,
            pred_b_b,
            pred_a_b,
            ], axis=1 )


        figure = numpy.concatenate( [ figure_A, figure_B ], axis=0 )
        figure = figure.reshape( (4,4) + figure.shape[1:] )
        figure = stack_images( figure )

        return numpy.clip( figure * 255, 0, 255 ).astype('uint8')

