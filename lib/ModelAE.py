# AutoEncoder base classes

import time
import numpy
from lib.training_data import TrainingDataGenerator, stack_images
from keras.losses import *


class ModelAE:
    def __init__(self, model_dir, filename_modifier):

        self.model_dir = model_dir
        self.encoder_filename = self.model_dir + '/encoder_' + filename_modifier + '.h5'
        self.decoder_A_filename = self.model_dir + '/decoder_A_' + filename_modifier + '.h5'
        self.decoder_B_filename = self.model_dir + '/decoder_B_' + filename_modifier + '.h5'

        self.encoder = self.Encoder()
        self.decoder_A = self.Decoder()
        self.decoder_B = self.Decoder()
        self.loss_function = mean_absolute_error
        self.initModel()
        
    #
    # set_loss_function has to be called *before* the model is initialized!
    #
    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def load(self, swapped):
        (decoder_A,decoder_B) = (self.decoder_A_filename, self.decoder_B_filename) if not swapped else (self.decoder_B_filename, self.decoder_A_filename)

        try:
            self.encoder.load_weights(self.encoder_filename)
            self.decoder_A.load_weights(decoder_A)
            self.decoder_B.load_weights(decoder_B)
            print('loaded model weights')
            return True
        except Exception as e:
            print('Failed loading existing training data.')
            print(e)
            return False

    def save_weights(self):
        self.encoder.save_weights(self.encoder_filename)
        self.decoder_A.save_weights(self.decoder_A_filename)
        self.decoder_B.save_weights(self.decoder_B_filename)
        print('saved model weights')

class TrainerAE():
    random_transform_args = {
        'rotation_range': 10,
        'zoom_range': 0.05,
        'shift_range': 0.05,
        'random_flip': 0.4,
    }

    def __init__(self, model, fn_A, fn_B, batch_size=64):
        self.batch_size = batch_size
        self.model = model

        generator = TrainingDataGenerator(self.random_transform_args, 160)
        self.images_A = generator.minibatchAB(fn_A, self.batch_size)
        self.images_B = generator.minibatchAB(fn_B, self.batch_size)

    def train_one_step(self, iter, viewer):
        epoch, warped_A, target_A = next(self.images_A)
        epoch, warped_B, target_B = next(self.images_B)

        loss_A = self.model.autoencoder_A.train_on_batch(warped_A, target_A)
        loss_B = self.model.autoencoder_B.train_on_batch(warped_B, target_B)
		
		# We need quite a bit of precision when printing the loss factors as the results of the squared loss functions can become quite small
        print("[{0}] [#{1:05d}] loss_A: {2:.7f}, loss_B: {3:.7f}".format(time.strftime("%H:%M:%S"), iter, loss_A, loss_B),
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

        figure = numpy.concatenate([figure_A, figure_B], axis=0)
        figure = figure.reshape((4, 7) + figure.shape[1:])
        figure = stack_images(figure)

        return numpy.clip(figure * 255, 0, 255).astype('uint8')
