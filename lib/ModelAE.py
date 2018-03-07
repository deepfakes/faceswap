# AutoEncoder base classes

import time
import numpy
from lib.training_data import TrainingDataGenerator, stack_images

encoderH5 = 'encoder.h5'
decoder_AH5 = 'decoder_A.h5'
decoder_BH5 = 'decoder_B.h5'

class ModelAE:
    def __init__(self, model_dir):
        self.model_dir = model_dir

        self.encoder = self.Encoder()
        self.decoder_A = self.Decoder()
        self.decoder_B = self.Decoder()

        self.initModel()

    def load(self, swapped):
        (face_A,face_B) = (decoder_AH5, decoder_BH5) if not swapped else (decoder_BH5, decoder_AH5)

        try:
            self.encoder.load_weights(str(self.model_dir / encoderH5))
            self.decoder_A.load_weights(str(self.model_dir / face_A))
            self.decoder_B.load_weights(str(self.model_dir / face_B))
            print('loaded model weights')
            return True
        except Exception as e:
            print('Failed loading existing training data.')
            print(e)
            return False

    def save_weights(self):
        self.encoder.save_weights(str(self.model_dir / encoderH5))
        self.decoder_A.save_weights(str(self.model_dir / decoder_AH5))
        self.decoder_B.save_weights(str(self.model_dir / decoder_BH5))
        print('saved model weights')

class TrainerAE():
    random_transform_args = {
        'rotation_range': 10,
        'zoom_range': 0.05,
        'shift_range': 0.05,
        'random_flip': 0.4,
    }

    def __init__(self, model, fn_A, fn_B, batch_size=64, balance_loss=False):
        self.batch_size = batch_size
        self.model = model

        generator = TrainingDataGenerator(self.random_transform_args, 160)
        self.images_A = generator.minibatchAB(fn_A, self.batch_size)
        self.images_B = generator.minibatchAB(fn_B, self.batch_size)
        
        self.balance_loss = balance_loss
        self.loss_A = 0.0
        self.loss_B = 0.0

    def train_one_step(self, iter, viewer):       

        if not self.balance_loss or self.loss_A == 0.0 or self.loss_B == 0.0 or iter % 10 == 0 or viewer is not None:
            is_train_A = True
            is_train_B = True
        else:
            is_train_A = self.loss_A >= self.loss_B
            is_train_B = not is_train_A
            
        if is_train_A:
            epoch, warped_A, target_A = next(self.images_A)
            self.loss_A = self.model.autoencoder_A.train_on_batch(warped_A, target_A)
            
        if is_train_B:
            epoch, warped_B, target_B = next(self.images_B)
            self.loss_B = self.model.autoencoder_B.train_on_batch(warped_B, target_B)       

        print("[{0}] [#{1:05d}] loss_A: {2:.5f}, loss_B: {3:.5f}".format(time.strftime("%H:%M:%S"), iter, self.loss_A, self.loss_B),
            end='\r')

        if viewer is not None:
            viewer(self.show_sample(target_A[0:14], target_B[0:14]), "training")

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

        if test_A.shape[0] % 2 == 1:
            figure_A = numpy.concatenate ([figure_A, numpy.expand_dims(figure_A[0],0) ])
            figure_B = numpy.concatenate ([figure_B, numpy.expand_dims(figure_B[0],0) ])
        
        figure = numpy.concatenate([figure_A, figure_B], axis=0)
        w = 4
        h = int( figure.shape[0] / w)        
        figure = figure.reshape((w, h) + figure.shape[1:])        
        figure = stack_images(figure)
        
        return numpy.clip(figure * 255, 0, 255).astype('uint8')
