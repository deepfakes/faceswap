import time
import numpy

from lib.training_data import TrainingDataGenerator, stack_images


TRANSFORM_PRC = 115.


class Trainer():
    
    _random_transform_args = {
        'rotation_range': 10 * (TRANSFORM_PRC * .01),
        'zoom_range': 0.05 * (TRANSFORM_PRC * .01),
        'shift_range': 0.05 * (TRANSFORM_PRC * .01),
        'random_flip': 0.4 * (TRANSFORM_PRC * .01),
    }
    
    def __init__(self, model, fn_A, fn_B, batch_size, *args):
        self.batch_size = batch_size
        self.model = model
        from timeit import default_timer as clock
        self._clock = clock
        
        generator = TrainingDataGenerator(self.random_transform_args, 160, 5, zoom=self.model.IMAGE_SHAPE[0]//64)        
        
        self.images_A = generator.minibatchAB(fn_A, self.batch_size)
        self.images_B = generator.minibatchAB(fn_B, self.batch_size)
                
        self.generator = generator        
        

    def train_one_step(self, iter_no, viewer):
        when = self._clock()
        _, warped_A, target_A = next(self.images_A)
        _, warped_B, target_B = next(self.images_B)

        loss_A = self.model.autoencoder_A.train_on_batch(warped_A, target_A)
        loss_B = self.model.autoencoder_B.train_on_batch(warped_B, target_B)
        
        self.model._epoch_no += 1        
                 
        if isinstance(loss_A, (list, tuple)):
            print("[{0}] [#{1:05d}] [{2:.3f}s] loss_A: {3:.5f}, loss_B: {4:.5f}".format(
                time.strftime("%H:%M:%S"), self.model._epoch_no, self._clock()-when, loss_A[1], loss_B[1]),
                end='\r')
        else:
            print("[{0}] [#{1:05d}] [{2:.3f}s] loss_A: {3:.5f}, loss_B: {4:.5f}".format(
                time.strftime("%H:%M:%S"), self.model._epoch_no, self._clock()-when, loss_A, loss_B),
                end='\r')         

        if viewer is not None:
            viewer(self.show_sample(target_A[0:8], target_B[0:8]), "training using {}, bs={}".format(self.model, self.batch_size))
            

    def show_sample(self, test_A, test_B):
        figure_A = numpy.stack([
            test_A,
            self.model.autoencoder_A.predict(test_A),
            self.model.autoencoder_B.predict(test_A),
        ], axis=1)
        
        figure_B = numpy.stack([
            test_B,
            self.model.autoencoder_B.predict(test_B),
            self.model.autoencoder_A.predict(test_B),
        ], axis=1)

        if (test_A.shape[0] % 2)!=0:
            figure_A = numpy.concatenate ([figure_A, numpy.expand_dims(figure_A[0],0) ])
            figure_B = numpy.concatenate ([figure_B, numpy.expand_dims(figure_B[0],0) ])

        figure = numpy.concatenate([figure_A, figure_B], axis=0)
        w = 4
        h = int( figure.shape[0] / w)
        figure = figure.reshape((w, h) + figure.shape[1:])
        figure = stack_images(figure)

        return numpy.clip(figure * 255, 0, 255).astype('uint8')
    
    
    @property
    def random_transform_args(self):
        return self._random_transform_args
