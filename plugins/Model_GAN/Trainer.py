import time
import cv2
import numpy as np

from lib.training_data import TrainingDataGenerator, stack_images

class GANTrainingDataGenerator(TrainingDataGenerator):
    def __init__(self, random_transform_args, coverage):
        super().__init__(random_transform_args, coverage)

    def color_adjust(self, img):
        return img / 255.0 * 2 - 1

class Trainer():
    random_transform_args = {
        'rotation_range': 20,
        'zoom_range': 0.05,
        'shift_range': 0.05,
        'random_flip': 0.5,
        }

    def __init__(self, model, fn_A, fn_B, batch_size):
        assert batch_size % 2 == 0, "batch_size must be an even number"
        self.batch_size = batch_size
        self.model = model
        
        self.use_mixup = True
        self.mixup_alpha = 0.2

        generator = GANTrainingDataGenerator(self.random_transform_args, 220)
        self.train_batchA = generator.minibatchAB(fn_A, batch_size)
        self.train_batchB = generator.minibatchAB(fn_B, batch_size)
    
    def train_one_step(self, iter, viewer):
        # ---------------------
        #  Train Discriminators
        # ---------------------

        # Select a random half batch of images
        epoch, warped_A, target_A = next(self.train_batchA) 
        epoch, warped_B, target_B = next(self.train_batchB) 

        # Generate a half batch of new images
        gen_alphasA, gen_imgsA = self.model.netGA.predict(warped_A)
        gen_alphasB, gen_imgsB = self.model.netGB.predict(warped_B)
        #gen_masked_imgsA = gen_alphasA * gen_imgsA + (1 - gen_alphasA) * warped_A
        #gen_masked_imgsB = gen_alphasB * gen_imgsB + (1 - gen_alphasB) * warped_B
        gen_masked_imgsA = np.array([gen_alphasA[i] * gen_imgsA[i] + (1 - gen_alphasA[i]) * warped_A[i] 
                                     for i in range(self.batch_size)])
        gen_masked_imgsB = np.array([gen_alphasB[i] * gen_imgsB[i] + (1 - gen_alphasB[i]) * warped_B[i]
                                     for i in range (self.batch_size)])

        valid = np.ones((self.batch_size, ) + self.model.netDA.output_shape[1:])
        fake = np.zeros((self.batch_size, ) + self.model.netDA.output_shape[1:])

        concat_real_inputA = np.array([np.concatenate([target_A[i], warped_A[i]], axis=-1) 
                                       for i in range(self.batch_size)])
        concat_real_inputB = np.array([np.concatenate([target_B[i], warped_B[i]], axis=-1) 
                                       for i in range(self.batch_size)])
        concat_fake_inputA = np.array([np.concatenate([gen_masked_imgsA[i], warped_A[i]], axis=-1) 
                                       for i in range(self.batch_size)])
        concat_fake_inputB = np.array([np.concatenate([gen_masked_imgsB[i], warped_B[i]], axis=-1) 
                                       for i in range(self.batch_size)])
        if self.use_mixup:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            mixup_A = lam * concat_real_inputA + (1 - lam) * concat_fake_inputA
            mixup_B = lam * concat_real_inputB + (1 - lam) * concat_fake_inputB

        # Train the discriminators
        #print ("Train the discriminators.")
        if self.use_mixup:
            d_lossA = self.model.netDA.train_on_batch(mixup_A, lam * valid)
            d_lossB = self.model.netDB.train_on_batch(mixup_B, lam * valid)
        else:
            d_lossA = self.model.netDA.train_on_batch(np.concatenate([concat_real_inputA, concat_fake_inputA], axis=0), 
                                                np.concatenate([valid, fake], axis=0))
            d_lossB = self.model.netDB.train_on_batch(np.concatenate([concat_real_inputB, concat_fake_inputB], axis=0),
                                                np.concatenate([valid, fake], axis=0))

        # ---------------------
        #  Train Generators
        # ---------------------

        # Train the generators
        #print ("Train the generators.")
        g_lossA = self.model.adversarial_autoencoderA.train_on_batch(warped_A, [target_A, valid])
        g_lossB = self.model.adversarial_autoencoderB.train_on_batch(warped_B, [target_B, valid])            
        
        print('[%s] [%d/%s][%d] Loss_DA: %f Loss_DB: %f Loss_GA: %f Loss_GB: %f'
              % (time.strftime("%H:%M:%S"), epoch, "num_epochs", iter, d_lossA[0], d_lossB[0], g_lossA[0], g_lossB[0]),
              end='\r')
        
        if viewer is not None:
            self.show_sample(viewer)
    
    def show_sample(self, display_fn):
        _, wA, tA = next(self.train_batchA)
        _, wB, tB = next(self.train_batchB)
        self.showG(tA, tB, display_fn)

    def showG(self, test_A, test_B, display_fn):
        def display_fig(name, figure_A, figure_B):
            figure = np.concatenate([figure_A, figure_B], axis=0 )
            columns = 4
            elements = figure.shape[0]
            figure = figure.reshape((columns,(elements//columns)) + figure.shape[1:])
            figure = stack_images(figure)
            figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
            display_fn(figure, name)

        out_test_A_netGA = self.model.netGA.predict(test_A)
        out_test_A_netGB = self.model.netGB.predict(test_A)
        out_test_B_netGA = self.model.netGA.predict(test_B)
        out_test_B_netGB = self.model.netGB.predict(test_B)

        figure_A = np.stack([
            test_A,
            out_test_A_netGA[1],
            out_test_A_netGB[1],
            ], axis=1 )
        figure_B = np.stack([
            test_B,
            out_test_B_netGB[1],
            out_test_B_netGA[1],
            ], axis=1 )
        
        display_fig("raw", figure_A, figure_B)       

        figure_A = np.stack([
            test_A,
            np.tile(out_test_A_netGA[0],3) * 2 - 1,
            np.tile(out_test_A_netGB[0],3) * 2 - 1,
            ], axis=1 )
        figure_B = np.stack([
            test_B,
            np.tile(out_test_B_netGB[0],3) * 2 - 1,
            np.tile(out_test_B_netGA[0],3) * 2 - 1,
            ], axis=1 )

        display_fig("alpha_masks", figure_A, figure_B)

        figure_A = np.stack([
            test_A,
            out_test_A_netGA[0] * out_test_A_netGA[1] + (1 - out_test_A_netGA[0]) * test_A,
            out_test_A_netGB[0] * out_test_A_netGB[1] + (1 - out_test_A_netGB[0]) * test_A,
            ], axis=1 )
        figure_B = np.stack([
            test_B,
            out_test_B_netGB[0] * out_test_B_netGB[1] + (1 - out_test_B_netGB[0]) * test_B,
            out_test_B_netGA[0] * out_test_B_netGA[1] + (1 - out_test_B_netGA[0]) * test_B,
            ], axis=1 )
        
        display_fig("masked", figure_A, figure_B)
