
import time
import numpy
from lib.training_data import TrainingDataGenerator, stack_images
from image_augmentation import random_transform, random_warp_src_dest

class DFTrainingDataGenerator(TrainingDataGenerator):
    def __init__(self, random_transform_args, coverage, scale, zoom):
        super().__init__(random_transform_args, coverage, scale, zoom)

    def minibatchAB(self, images, srcPoints, dstPoints, batch_size ):
        batch = BackgroundGenerator(self.minibatch(images, srcPoints, dstPoints, batch_size), 1)
        for ep1, warped_img, target_img, mask_img in batch.iterator():
            yield ep1, warped_img, target_img, mask_img

    def minibatch(self, data, srcPoints, dstPoints, batchsize):
        epoch = 0
        while True:
            epoch+=1
            indices = numpy.random.choice(range(0,images.shape[0]),size=batch_size,replace=False)
            for i,index in enumerate(indices):
                image = images[index]
                image = random_transform( image, **random_transform_args )

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

            yield epoch, warped_img, target_img, mask_image


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

        generator = TrainingDataGenerator(self.random_transform_args, 160)
        self.images_A = generator.minibatchAB(fn_A, landmarks_A, landmarks_B, self.batch_size)
        self.images_B = generator.minibatchAB(fn_B, landmarks_B, landmarks_A, self.batch_size)

    def train_one_step(self, iter, viewer):
        epoch, warped_A, target_A, mask_A = next(self.images_A)
        epoch, warped_B, target_B, mask_B = next(self.images_B)

        #missing normalisation:   images_A[:,:,:3] += images_B[:,:,:3].mean( axis=(0,1,2) ) - images_A[:,:,:3].mean( axis=(0,1,2) )
      
        #omask = numpy.ones((target_A.shape[0],64,64,1),float)

        loss_A = autoencoder_A.train_on_batch([warped_A,mask_A], [target_A,mask_A])
        loss_B = autoencoder_B.train_on_batch([warped_B,mask_B], [target_B,mask_B])
        print("[{0}] [#{1:05d}] loss_A: {2:.5f}, loss_B: {3:.5f}".format(time.strftime("%H:%M:%S"), iter, loss_A, loss_B),
            end='\r')

        if viewer is not None:
            viewer(self.show_sample(target_A[0:8,:,:,:3], target_B[0:8,:,:,:3]), "training")

    def show_sample(self, test_A, test_B):
        test_A_i = []
        test_B_i = []
        
        for i in test_A:
            test_A_i.append(cv2.resize(i,(64,64),cv2.INTER_AREA))
        test_A_i = numpy.array(test_A_i).reshape((-1,64,64,3))

        for i in test_B:
            test_B_i.append(cv2.resize(i,(64,64),cv2.INTER_AREA))
        test_B_i = numpy.array(test_B_i).reshape((-1,64,64,3))

        figWarped = numpy.stack([warped_A[:6],warped_B[:6]],axis=0 )
        figWarped = numpy.clip( figWarped * 255, 0, 255 ).astype('uint8')
        figWarped = stack_images( figWarped )
        cv2.imshow( "w", figWarped )
        
        zmask = numpy.zeros((test_A.shape[0],128,128,1),float)

        pred_a_a,pred_a_a_m = autoencoder_A.predict([test_A_i,zmask])
        pred_b_a,pred_b_a_m = autoencoder_B.predict([test_A_i,zmask])

        pred_a_b,pred_a_b_m = autoencoder_A.predict([test_B_i,zmask])
        pred_b_b,pred_b_b_m = autoencoder_B.predict([test_B_i,zmask])

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

