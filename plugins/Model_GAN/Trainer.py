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

        self.lrD = 2e-4 # Discriminator learning rate
        self.lrG = 2e-4 # Generator learning rate

        generator = GANTrainingDataGenerator(self.random_transform_args, 220, 6, 1)
        self.train_batchA = generator.minibatchAB(fn_A, batch_size)
        self.train_batchB = generator.minibatchAB(fn_B, batch_size)

        self.avg_counter = self.errDA_sum = self.errDB_sum = self.errGA_sum = self.errGB_sum = self.errDA2_sum = self.errDB2_sum = self.errDA_code_sum = self.errDB_code_sum = 0

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
        self.loss_fn_bce = lambda output, target : -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target))

        # ========== Define Perceptual Loss Model==========
        if self.use_perceptual_loss:
            from keras.models import Model
            from keras_vggface.vggface import VGGFace
            print("Using perceptual loss.")
            vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))
            vggface.trainable = False
            out_size55 = vggface.layers[36].output
            out_size28 = vggface.layers[78].output
            out_size7 = vggface.layers[-2].output
            vggface_feat = Model(vggface.input, [out_size55, out_size28, out_size7])
            vggface_feat.trainable = False
            netDA_feat = netDB_feat = 0
        else:
            print("Not using perceptual loss.")
            vggface_feat = None
            netDA_feat = netDB_feat = vggface_feat = None

        #TODO check "Tips for mask refinement (optional after >15k iters)" => https://render.githubusercontent.com/view/ipynb?commit=87d6e7a28ce754acd38d885367b6ceb0be92ec54&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f7368616f616e6c752f66616365737761702d47414e2f383764366537613238636537353461636433386438383533363762366365623062653932656335342f46616365537761705f47414e5f76325f737a3132385f747261696e2e6970796e62&nwo=shaoanlu%2Ffaceswap-GAN&path=FaceSwap_GAN_v2_sz128_train.ipynb&repository_id=115182783&repository_type=Repository#Tips-for-mask-refinement-(optional-after-%3E15k-iters)
        loss_DA, loss_DA2, loss_GA, loss_DA_feat, loss_DA_code = self.define_loss(self.model.netDA, self.model.netDA2, netDA_feat, self.model.netD_code, self.model.netGA, real_A, fake_A, distorted_A, "A", vggface_feat)
        loss_DB, loss_DB2, loss_GB, loss_DB_feat, loss_DB_code = self.define_loss(self.model.netDB, self.model.netDB2, netDB_feat, self.model.netD_code, self.model.netGB, real_B, fake_B, distorted_B, "B", vggface_feat)

        loss_GA += 1e-3 * K.mean(K.abs(mask_A))
        loss_GB += 1e-3 * K.mean(K.abs(mask_B))

        w_fo = 0.01
        loss_GA += w_fo * K.mean(self.first_order(mask_A, axis=1))
        loss_GA += w_fo * K.mean(self.first_order(mask_A, axis=2))
        loss_GB += w_fo * K.mean(self.first_order(mask_B, axis=1))
        loss_GB += w_fo * K.mean(self.first_order(mask_B, axis=2))

        weightsDA = self.model.netDA.trainable_weights
        weightsDA2 = self.model.netDA2.trainable_weights
        weightsGA = self.model.netGA.trainable_weights
        weightsDB = self.model.netDB.trainable_weights
        weightsDB2 = self.model.netDB2.trainable_weights
        weightsGB = self.model.netGB.trainable_weights
        weightsD_code = self.model.netD_code.trainable_weights

        # Adam(..).get_updates(...)
        """
        # Using the following update function spped up training time (per iter.) by ~15%.
        training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsDA+weightsDA2+weightsD_code,[],loss_DA+loss_DA2+loss_DA_code)
        netDA_train = K.function([distorted_A, real_A],[loss_DA+loss_DA2+loss_DA_code], training_updates)
        """
        training_updates = Adam(lr=self.lrD, beta_1=0.5).get_updates(weightsDA,[],loss_DA)
        self.netDA_train = K.function([distorted_A, real_A],[loss_DA], training_updates)
        training_updates = Adam(lr=self.lrD, beta_1=0.5).get_updates(weightsDA2,[],loss_DA2)
        self.netDA2_train = K.function([distorted_A, real_A],[loss_DA2], training_updates)
        training_updates = Adam(lr=self.lrG, beta_1=0.5).get_updates(weightsGA,[], loss_GA)
        self.netGA_train = K.function([distorted_A, real_A], [loss_GA], training_updates)

        training_updates = Adam(lr=self.lrD, beta_1=0.5).get_updates(weightsDB,[],loss_DB)
        self.netDB_train = K.function([distorted_B, real_B],[loss_DB], training_updates)
        training_updates = Adam(lr=self.lrD, beta_1=0.5).get_updates(weightsDB2,[],loss_DB2)
        self.netDB2_train = K.function([distorted_B, real_B],[loss_DB2], training_updates)
        training_updates = Adam(lr=self.lrG, beta_1=0.5).get_updates(weightsGB,[], loss_GB)
        self.netGB_train = K.function([distorted_B, real_B], [loss_GB], training_updates)

        training_updates = Adam(lr=self.lrD, beta_1=0.5).get_updates(weightsD_code,[], loss_DA_code)
        self.netDA_code_train = K.function([distorted_A, real_A],[loss_DA_code], training_updates)
        training_updates = Adam(lr=self.lrD, beta_1=0.5).get_updates(weightsD_code,[], loss_DB_code)
        self.netDB_code_train = K.function([distorted_B, real_B],[loss_DB_code], training_updates)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html
    def cos_distance(self, x1, x2):
        x1 = K.l2_normalize(x1, axis=-1)
        x2 = K.l2_normalize(x2, axis=-1)
        return K.mean(1 - K.sum((x1 * x2), axis=-1))

    def first_order(self, x, axis=1):
        img_nrows = x.shape[1]
        img_ncols = x.shape[2]
        if axis == 1:
            return K.abs(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        elif axis == 2:
            return K.abs(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
        else:
            return None

    def train_one_step(self, iter, viewer, save_interval):
        # ---------------------
        #  Train Discriminators
        # ---------------------

        # Select a random half batch of images
        epoch, warped_A, target_A = next(self.train_batchA)
        epoch, warped_B, target_B = next(self.train_batchB)

        # Train dicriminators for one batch
        errDA  = self.netDA_train([warped_A, target_A])
        errDB  = self.netDB_train([warped_B, target_B])
        errDA2 = self.netDA2_train([warped_A, target_A])
        errDB2 = self.netDB2_train([warped_B, target_B])
        errDA_code = self.netDA_code_train([warped_A, target_A])
        errDB_code = self.netDB_code_train([warped_B, target_B])

        # Train generators for one batch
        errGA = self.netGA_train([warped_A, target_A])
        errGB = self.netGB_train([warped_B, target_B])

        # For calculating average losses
        self.errDA_sum += errDA[0]
        self.errDB_sum += errDB[0]
        self.errGA_sum += errGA[0]
        self.errGB_sum += errGB[0]
        self.errDA2_sum += errDA2[0]
        self.errDB2_sum += errDB2[0]
        self.errDA_code_sum += errDA_code[0]
        self.errDB_code_sum += errDB_code[0]
        self.avg_counter += 1

        errDA_avg = self.errDA_sum/self.avg_counter
        errDB_avg = self.errDB_sum/self.avg_counter
        errDA2_avg = self.errDA2_sum/self.avg_counter
        errDB2_avg = self.errDB2_sum/self.avg_counter
        errDA_code_avg = self.errDA_code_sum/self.avg_counter
        errDB_code_avg = self.errDB_code_sum/self.avg_counter
        errGA_avg = self.errGA_sum/self.avg_counter
        errGB_avg = self.errGB_sum/self.avg_counter

        if (int(iter/save_interval) % (save_interval*15) == 0 and iter/save_interval > 0) or iter == 0:
            print('Losses:                     {}DA,       DB,       DA2,      DB2,      DA_code,  DB_code,  GA,       GB'.format(' '*len(str(iter))))

        print('[%s] [%d/%s][%d] %f, %f, %f, %f, %f, %f, %f, %f'
              % (time.strftime("%H:%M:%S"), epoch, "num_epochs", iter, errDA_avg, errDB_avg, errDA2_avg, errDB2_avg, errDA_code_avg, errDB_code_avg, errGA_avg, errGB_avg),
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

    def define_loss(self, netD, netD2, netD_feat, netD_code, netG, real, fake_argb, distorted, domain, vggface_feat=None):
        alpha = Lambda(lambda x: x[:,:,:, :1])(fake_argb)
        fake_rgb = Lambda(lambda x: x[:,:,:, 1:])(fake_argb)
        fake = alpha * fake_rgb + (1-alpha) * distorted

        # Use mixup - Loss of masked output
        dist = Beta(self.mixup_alpha, self.mixup_alpha)
        lam = dist.sample()
        # ==========
        mixup = lam * concatenate([real, distorted]) + (1 - lam) * concatenate([fake, distorted])
        # ==========
        output_mixup = netD(mixup)
        loss_D = self.loss_fn(output_mixup, lam * K.ones_like(output_mixup))
        loss_G = .5 * self.loss_fn(output_mixup, (1 - lam) * K.ones_like(output_mixup))
        # Loss of raw output
        #real_shuffled = Lambda(lambda x: tf.random_shuffle(x))(real)
        lam2 = dist.sample()
        mixup2 = lam2 * real + (1 - lam2) * fake_rgb
        output2_mixup = netD2(mixup2)
        loss_D2 = self.loss_fn(output2_mixup, lam2 * K.ones_like(output2_mixup))
        loss_G += .5 * self.loss_fn(output2_mixup, (1 - lam) * K.ones_like(output2_mixup))

        # Domain adversarial loss
        real_code = netG([real])[1]
        rec_code = netG([fake_rgb])[1]
        output_real_code = netD_code([real_code])
        # Target of domain A = 1, domain B = 0
        if domain == "A":
            loss_D_code = self.loss_fn_bce(output_real_code, K.ones_like(output_real_code))
            loss_G += .03 * self.loss_fn(output_real_code, K.zeros_like(output_real_code))
        elif domain == "B":
            loss_D_code = self.loss_fn_bce(output_real_code, K.zeros_like(output_real_code))
            loss_G += .03 * self.loss_fn(output_real_code, K.ones_like(output_real_code))

        # semantic consistency loss
        loss_G += 1. * self.cos_distance(rec_code, real_code)

        # ==========
        # L1 loss
        loss_G += 3 * K.mean(K.abs(fake_rgb - real))
        # ==========

        loss_D_feat = 0
        # Perceptual Loss
        if not vggface_feat is None:
            def preprocess_vggface(x):
                x = (x + 1)/2 * 255 # channel order: BGR
                x -= [93.5940, 104.7624, 129.]
                return x
            pl_params = (0.02, 0.3, 0.5)
            real_sz224 = tf.image.resize_images(real, [224, 224])
            real_sz224 = Lambda(preprocess_vggface)(real_sz224)
            # ==========
            fake_sz224 = tf.image.resize_images(fake_rgb, [224, 224])
            fake_sz224 = Lambda(preprocess_vggface)(fake_sz224)
            # ==========
            real_feat55, real_feat28, real_feat7 = vggface_feat(real_sz224)
            fake_feat55, fake_feat28, fake_feat7  = vggface_feat(fake_sz224)
            loss_G += pl_params[0] * K.mean(K.square(fake_feat7 - real_feat7))
            loss_G += pl_params[1] * K.mean(K.abs(fake_feat28 - real_feat28))
            loss_G += pl_params[2] * K.mean(K.abs(fake_feat55 - real_feat55))

        return loss_D, loss_D2, loss_G, loss_D_feat, loss_D_code

    def show_sample(self, display_fn):
        _, wA, tA = next(self.train_batchA)
        _, wB, tB = next(self.train_batchB)
        display_fn(self.showG(tA, tB, self.path_A, self.path_B), "raw")
        display_fn(self.showG(tA, tB, self.path_bgr_A, self.path_bgr_B), "masked")
        display_fn(self.showG_mask(tA, tB, self.path_mask_A, self.path_mask_B), "mask")
        # Reset the averages
        self.errDA_sum = self.errDB_sum = self.errDA2_sum = self.errDB2_sum = self.errDA_code_sum = self.errDB_code_sum = self.errGA_sum = self.errGB_sum = 0
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
