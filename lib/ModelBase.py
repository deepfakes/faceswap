import numpy
from lib.training_data import minibatchAB

class ModelBase:
    def __init__(self, model_dir):

        self.model_dir = model_dir

        self.encoder = self.Encoder()
        self.decoder_A = self.Decoder()
        self.decoder_B = self.Decoder()

        self.initModel()

    def load(self, swapped):
        (face_A,face_B) = ('/decoder_A.h5', '/decoder_B.h5') if not swapped else ('/decoder_B.h5', '/decoder_A.h5')

        try:
            self.encoder.load_weights(self.model_dir + '/encoder.h5')
            self.decoder_A.load_weights(self.model_dir + face_A)
            self.decoder_B.load_weights(self.model_dir + face_B)
            print('loaded model weights')
        except Exception as e:
            print('Failed loading existing training data.')
            print(e)

    def save_weights(self):
        self.encoder.save_weights(self.model_dir + '/encoder.h5')
        self.decoder_A.save_weights(self.model_dir + '/decoder_A.h5')
        self.decoder_B.save_weights(self.model_dir + '/decoder_B.h5')
        print('saved model weights')

class TrainerBase():
    BATCH_SIZE = 64

    def __init__(self, model, fn_A, fn_B):
        self.model = model
        self.images_A = minibatchAB(fn_A, self.BATCH_SIZE)
        self.images_B = minibatchAB(fn_B, self.BATCH_SIZE)

    def train_one_step(self, iter):
        epoch, warped_A, target_A = next(self.images_A)
        epoch, warped_B, target_B = next(self.images_B)

        loss_A = self.model.autoencoder_A.train_on_batch(warped_A, target_A)
        loss_B = self.model.autoencoder_B.train_on_batch(warped_B, target_B)
        print(loss_A, loss_B)

        return lambda: self.show_sample(target_A[0:14], target_B[0:14])

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
