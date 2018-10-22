import time
import cv2
import numpy as np

from keras.layers import *
from tensorflow.contrib.distributions import Beta
import tensorflow as tf
from keras.optimizers import Adam
from keras import backend as K

from lib.training_data import TrainingDataGenerator, stack_images

class GANTrainingDataGenerator(TrainingDataGenerator):
    def __init__(self, random_transform_args, coverage, scale, zoom):
        super().__init__(random_transform_args, coverage, scale, zoom)

    def color_adjust(self, img):
        return img / 255.0 * 2 - 1

class Trainer():
    random_transform_args = {
        'rotation_range': 20,
        'zoom_range': 0.1,
        'shift_range': 0.05,
        'random_flip': 0.5,
        }

    def __init__(self, model, fn_A, fn_B, batch_size, perceptual_loss):
        K.set_learning_phase(1)

        assert batch_size % 2 == 0, "batch_size must be an even number"
        self.batch_size = batch_size
        self.model = model

        self.use_lsgan = True
        self.use_mixup = True
        self.mixup_alpha = 0.2
        self.use_perceptual_loss = perceptual_loss
        self.use_instancenorm = False

        self.lrD = 1e-4 # Discriminator learning rate
        self.lrG = 1e-4 # Generator learning rate

        generator = GANTrainingDataGenerator(self.random_transform_args, 220, 6, 1)
        self.train_batchA = generator.minibatchAB(fn_A, batch_size)
        self.train_batchB = generator.minibatchAB(fn_B, batch_size)

        self.avg_counter = self.errDA_sum = self.errDB_sum = self.errGA_sum = self.errGB_sum = 0

        self.setup()

    def setup(self):
        distorted_A, fake_A, mask_A, self.path_A, self.path_mask_A, self.path_abgr_A, self.path_bgr_A = self.cycle_variables(self.model.netGA)
        distorted_B, fake_B, mask_B, self.path_B, self.path_mask_B, self.path_abgr_B, self.path_bgr_B = self.cycle_variables(self.model.netGB)
        real_A = Input(shape=self.model.img_shape)
        real_B = Input(shape=self.model.img_shape)

        if self.use_lsgan:
            self.loss_fn = lambda output, target : K.mean(K.abs(K.square(output-target)))
        else:
            self.loss_fn = lambda output, target : -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target))

        # ========== Define Perceptual Loss Model==========
        if self.use_perceptual_loss:
            from keras.models import Model
            from keras_vggface.vggface import VGGFace
            vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))
            vggface.trainable = False
            out_size55 = vggface.layers[36].output
            out_size28 = vggface.layers[78].output
            out_size7 = vggface.layers[-2].output
            vggface_feat = Model(vggface.input, [out_size55, out_size28, out_size7])
            vggface_feat.trainable = False
        else:
            vggface_feat = None

        #TODO check "Tips for mask refinement (optional after >15k iters)" => https://render.githubusercontent.com/view/ipynb?commit=87d6e7a28ce754acd38d885367b6ceb0be92ec54&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f7368616f616e6c752f66616365737761702d47414e2f383764366537613238636537353461636433386438383533363762366365623062653932656335342f46616365537761705f47414e5f76325f737a3132385f747261696e2e6970796e62&nwo=shaoanlu%2Ffaceswap-GAN&path=FaceSwap_GAN_v2_sz128_train.ipynb&repository_id=115182783&repository_type=Repository#Tips-for-mask-refinement-(optional-after-%3E15k-iters)
        loss_DA, loss_GA = self.define_loss(self.model.netDA, real_A, fake_A, distorted_A, vggface_feat)
        loss_DB, loss_GB = self.define_loss(self.model.netDB, real_B, fake_B, distorted_B, vggface_feat)

        loss_GA += 1e-3 * K.mean(K.abs(mask_A))
        loss_GB += 1e-3 * K.mean(K.abs(mask_B))

        w_fo = 0.01
        loss_GA += w_fo * K.mean(self.first_order(mask_A, axis=1))
        loss_GA += w_fo * K.mean(self.first_order(mask_A, axis=2))
        loss_GB += w_fo * K.mean(self.first_order(mask_B, axis=1))
        loss_GB += w_fo * K.mean(self.first_order(mask_B, axis=2))

        weightsDA = self.model.netDA.trainable_weights
        weightsGA = self.model.netGA.trainable_weights
        weightsDB = self.model.netDB.trainable_weights
        weightsGB = self.model.netGB.trainable_weights

        # Adam(..).get_updates(...)
        training_updates = Adam(lr=self.lrD, beta_1=0.5).get_updates(weightsDA,[],loss_DA)
        self.netDA_train = K.function([distorted_A, real_A],[loss_DA], training_updates)
        training_updates = Adam(lr=self.lrG, beta_1=0.5).get_updates(weightsGA,[], loss_GA)
        self.netGA_train = K.function([distorted_A, real_A], [loss_GA], training_updates)

        training_updates = Adam(lr=self.lrD, beta_1=0.5).get_updates(weightsDB,[],loss_DB)
        self.netDB_train = K.function([distorted_B, real_B],[loss_DB], training_updates)
        training_updates = Adam(lr=self.lrG, beta_1=0.5).get_updates(weightsGB,[], loss_GB)
        self.netGB_train = K.function([distorted_B, real_B], [loss_GB], training_updates)

    def first_order(self, x, axis=1):
        img_nrows = x.shape[1]
        img_ncols = x.shape[2]
        if axis == 1:
            return K.abs(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        elif axis == 2:
            return K.abs(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
        else:
            return None

    def train_one_step(self, iter, viewer):
        # ---------------------
        #  Train Discriminators
        # ---------------------

        # Select a random half batch of images
        epoch, warped_A, target_A = next(self.train_batchA)
        epoch, warped_B, target_B = next(self.train_batchB)

        # Train dicriminators for one batch
        errDA  = self.netDA_train([warped_A, target_A])
        errDB  = self.netDB_train([warped_B, target_B])

        # Train generators for one batch
        errGA = self.netGA_train([warped_A, target_A])
        errGB = self.netGB_train([warped_B, target_B])

        # For calculating average losses
        self.errDA_sum += errDA[0]
        self.errDB_sum += errDB[0]
        self.errGA_sum += errGA[0]
        self.errGB_sum += errGB[0]
        self.avg_counter += 1

        print('[%s] [%d/%s][%d] Loss_DA: %f Loss_DB: %f Loss_GA: %f Loss_GB: %f'
              % (time.strftime("%H:%M:%S"), epoch, "num_epochs", iter, self.errDA_sum/self.avg_counter, self.errDB_sum/self.avg_counter, self.errGA_sum/self.avg_counter, self.errGB_sum/self.avg_counter),
              end='\r')

        if viewer is not None:
            self.show_sample(viewer)

    def cycle_variables(self, netG):
        distorted_input = netG.inputs[0]
        fake_output = netG.outputs[0]
        alpha = Lambda(lambda x: x[:,:,:, :1])(fake_output)
        rgb = Lambda(lambda x: x[:,:,:, 1:])(fake_output)

        masked_fake_output = alpha * rgb + (1-alpha) * distorted_input

        fn_generate = K.function([distorted_input], [masked_fake_output])
        fn_mask = K.function([distorted_input], [concatenate([alpha, alpha, alpha])])
        fn_abgr = K.function([distorted_input], [concatenate([alpha, rgb])])
        fn_bgr = K.function([distorted_input], [rgb])
        return distorted_input, fake_output, alpha, fn_generate, fn_mask, fn_abgr, fn_bgr

    def define_loss(self, netD, real, fake_argb, distorted, vggface_feat=None):
        alpha = Lambda(lambda x: x[:,:,:, :1])(fake_argb)
        fake_rgb = Lambda(lambda x: x[:,:,:, 1:])(fake_argb)
        fake = alpha * fake_rgb + (1-alpha) * distorted

        if self.use_mixup:
            dist = Beta(self.mixup_alpha, self.mixup_alpha)
            lam = dist.sample()
            # ==========
            mixup = lam * concatenate([real, distorted]) + (1 - lam) * concatenate([fake, distorted])
            # ==========
            output_mixup = netD(mixup)
            loss_D = self.loss_fn(output_mixup, lam * K.ones_like(output_mixup))
            output_fake = netD(concatenate([fake, distorted])) # dummy
            loss_G = .5 * self.loss_fn(output_mixup, (1 - lam) * K.ones_like(output_mixup))
        else:
            output_real = netD(concatenate([real, distorted])) # positive sample
            output_fake = netD(concatenate([fake, distorted])) # negative sample
            loss_D_real = self.loss_fn(output_real, K.ones_like(output_real))
            loss_D_fake = self.loss_fn(output_fake, K.zeros_like(output_fake))
            loss_D = loss_D_real + loss_D_fake
            loss_G = .5 * self.loss_fn(output_fake, K.ones_like(output_fake))
        # ==========
        loss_G += K.mean(K.abs(fake_rgb - real))
        # ==========

        # Edge loss (similar with total variation loss)
        loss_G += 1 * K.mean(K.abs(self.first_order(fake_rgb, axis=1) - self.first_order(real, axis=1)))
        loss_G += 1 * K.mean(K.abs(self.first_order(fake_rgb, axis=2) - self.first_order(real, axis=2)))


        # Perceptual Loss
        if not vggface_feat is None:
            def preprocess_vggface(x):
                x = (x + 1)/2 * 255 # channel order: BGR
                #x[..., 0] -= 93.5940
                #x[..., 1] -= 104.7624
                #x[..., 2] -= 129.
                x -= [91.4953, 103.8827, 131.0912]
                return x
            pl_params = (0.011, 0.11, 0.1919)
            real_sz224 = tf.image.resize_images(real, [224, 224])
            real_sz224 = Lambda(preprocess_vggface)(real_sz224)
            # ==========
            fake_sz224 = tf.image.resize_images(fake_rgb, [224, 224])
            fake_sz224 = Lambda(preprocess_vggface)(fake_sz224)
            # ==========
            real_feat55, real_feat28, real_feat7 = vggface_feat(real_sz224)
            fake_feat55, fake_feat28, fake_feat7  = vggface_feat(fake_sz224)
            loss_G += pl_params[0] * K.mean(K.abs(fake_feat7 - real_feat7))
            loss_G += pl_params[1] * K.mean(K.abs(fake_feat28 - real_feat28))
            loss_G += pl_params[2] * K.mean(K.abs(fake_feat55 - real_feat55))

        return loss_D, loss_G

    def show_sample(self, display_fn):
        _, wA, tA = next(self.train_batchA)
        _, wB, tB = next(self.train_batchB)
        display_fn(self.showG(tA, tB, self.path_A, self.path_B), "masked")
        display_fn(self.showG(tA, tB, self.path_bgr_A, self.path_bgr_B), "raw")
        display_fn(self.showG_mask(tA, tB, self.path_mask_A, self.path_mask_B), "mask")
        # Reset the averages
        self.errDA_sum = self.errDB_sum = self.errGA_sum = self.errGB_sum = 0
        self.avg_counter = 0

    def showG(self, test_A, test_B, path_A, path_B):
        figure_A = np.stack([
            test_A,
            np.squeeze(np.array([path_A([test_A[i:i+1]]) for i in range(test_A.shape[0])])),
            np.squeeze(np.array([path_B([test_A[i:i+1]]) for i in range(test_A.shape[0])])),
            ], axis=1 )
        figure_B = np.stack([
            test_B,
            np.squeeze(np.array([path_B([test_B[i:i+1]]) for i in range(test_B.shape[0])])),
            np.squeeze(np.array([path_A([test_B[i:i+1]]) for i in range(test_B.shape[0])])),
            ], axis=1 )

        figure = np.concatenate([figure_A, figure_B], axis=0 )
        figure = figure.reshape((4,self.batch_size // 2) + figure.shape[1:])
        figure = stack_images(figure)
        figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
        return figure

    def showG_mask(self, test_A, test_B, path_A, path_B):
        figure_A = np.stack([
            test_A,
            (np.squeeze(np.array([path_A([test_A[i:i+1]]) for i in range(test_A.shape[0])])))*2-1,
            (np.squeeze(np.array([path_B([test_A[i:i+1]]) for i in range(test_A.shape[0])])))*2-1,
            ], axis=1 )
        figure_B = np.stack([
            test_B,
            (np.squeeze(np.array([path_B([test_B[i:i+1]]) for i in range(test_B.shape[0])])))*2-1,
            (np.squeeze(np.array([path_A([test_B[i:i+1]]) for i in range(test_B.shape[0])])))*2-1,
            ], axis=1 )

        figure = np.concatenate([figure_A, figure_B], axis=0 )
        figure = figure.reshape((4,self.batch_size // 2) + figure.shape[1:])
        figure = stack_images(figure)
        figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
        return figure
