# Improved-AutoEncoder base classes

import time
import numpy
from lib.training_data import TrainingDataGenerator, stack_images

encoderH5 = 'encoder.h5'
decoderH5 = 'decoder.h5'
inter_AH5 = 'inter_A.h5'
inter_BH5 = 'inter_B.h5'
inter_bothH5 = 'inter_both.h5'

class ModelIAE:
    def __init__(self, model_dir):
        self.model_dir = model_dir

        self.encoder = self.Encoder()
        self.decoder = self.Decoder()
        self.inter_A = self.Intermidiate()
        self.inter_B = self.Intermidiate()
        self.inter_both = self.Intermidiate()

        self.initModel()

    def load(self, swapped):
        (face_A,face_B) = (inter_AH5, inter_BH5) if not swapped else (inter_AH5, inter_BH5)

        try:
            self.encoder.load_weights(str(self.model_dir / encoderH5))
            self.decoder.load_weights(str(self.model_dir / decoderH5))
            self.inter_both.load_weights(str(self.model_dir / inter_bothH5))
            self.inter_A.load_weights(str(self.model_dir / face_A))
            self.inter_B.load_weights(str(self.model_dir / face_B))
            print('loaded model weights')
            return True
        except Exception as e:
            print('Failed loading existing training data.')
            print(e)
            return False

    def save_weights(self):
        self.encoder.save_weights(str(self.model_dir / encoderH5))
        self.decoder.save_weights(str(self.model_dir / decoderH5))
        self.inter_both.save_weights(str(self.model_dir / inter_bothH5))
        self.inter_A.save_weights(str(self.model_dir / inter_AH5))
        self.inter_B.save_weights(str(self.model_dir / inter_BH5))
        print('saved model weights')

class TrainerIAE():
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
        print("[{0}] [#{1:05d}] loss_A: {2:.5f}, loss_B: {3:.5f}".format(time.strftime("%H:%M:%S"), iter, loss_A, loss_B),
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
