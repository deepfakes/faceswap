import cv2
import numpy

from lib.utils import get_image_paths, load_images, stack_images
from lib.training_data import get_training_data

from lib.model import autoencoder_A
from lib.model import autoencoder_B
from lib.model import encoder, decoder_A, decoder_B

try:
    encoder.load_weights('models/encoder.h5')
    decoder_A.load_weights('models/decoder_A.h5')
    decoder_B.load_weights('models/decoder_B.h5')
except:
    pass


def save_model_weights():
    encoder.save_weights('models/encoder.h5')
    decoder_A.save_weights('models/decoder_A.h5')
    decoder_B.save_weights('models/decoder_B.h5')
    print('save model weights')


def show_sample(test_A, test_B):
    figure_A = numpy.stack([
        test_A,
        autoencoder_A.predict(test_A),
        autoencoder_B.predict(test_A),
        ], axis=1)
    figure_B = numpy.stack([
        test_B,
        autoencoder_B.predict(test_B),
        autoencoder_A.predict(test_B),
        ], axis=1)

    figure = numpy.concatenate([figure_A, figure_B], axis=0)
    figure = figure.reshape((4, 7) + figure.shape[1:])
    figure = stack_images(figure)

    figure = numpy.clip(figure * 255, 0, 255).astype('uint8')

    cv2.imwrite('_sample.jpg', figure)


images_A = get_image_paths('data/trump')
images_B = get_image_paths('data/cage')
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0

images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))

print('press "q" to stop training and save model')

BATCH_SIZE = 64

for epoch in range(1000000):
    warped_A, target_A = get_training_data(images_A, BATCH_SIZE)
    warped_B, target_B = get_training_data(images_B, BATCH_SIZE)

    loss_A = autoencoder_A.train_on_batch(warped_A, target_A)
    loss_B = autoencoder_B.train_on_batch(warped_B, target_B)
    print(loss_A, loss_B)

    if epoch % 100 == 0:
        save_model_weights()
        show_sample(target_A[0:14], target_B[0:14])

    key = cv2.waitKey(1)
    if key == ord('q'):
        save_model_weights()
        exit()

