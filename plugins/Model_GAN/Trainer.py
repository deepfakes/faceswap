import time
import cv2
import numpy as np

from lib.training_data import minibatchAB
from .Trainable import Trainable

display_iters = 50
niter = 150

class Trainer():
    BATCH_SIZE = 32

    errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
    gen_iterations = 0

    def __init__(self, model, fn_A, fn_B):
        self.model = model
        self.images_A = minibatchAB(fn_A, self.BATCH_SIZE)
        self.images_B = minibatchAB(fn_B, self.BATCH_SIZE)

        self.trainer_A = Trainable(model.netGA, model.netDA, model.IMAGE_SHAPE)
        self.trainer_B = Trainable(model.netGB, model.netDB, model.IMAGE_SHAPE)

    def train_one_step(self, iter):
        epoch, warped_A, target_A = next(self.images_A)
        epoch, warped_B, target_B = next(self.images_B)

        # Train dicriminators for one batch
        if iter % 1 == 0:
            errDA = self.trainer_A.trainD([warped_A, target_A])
            errDB = self.trainer_B.trainD([warped_B, target_B])

        # Train generators for one batch
        errGA = self.trainer_A.trainG([warped_A, target_A])
        errGB = self.trainer_B.trainG([warped_B, target_B])

        self.errDA_sum += errDA[0]
        self.errDB_sum += errDB[0]
        self.errGA_sum += errGA[0]
        self.errGB_sum += errGB[0]
        
        if self.gen_iterations % display_iters == 0:
            print("[%s] [%d/%d][%d] Loss_DA: %f Loss_DB: %f Loss_GA: %f Loss_GB: %f" % (time.strftime("%H:%M:%S"), iter, niter, self.gen_iterations, self.errDA_sum/display_iters, self.errDB_sum/display_iters, self.errGA_sum/display_iters, self.errGB_sum/display_iters))
            errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
        elif self.gen_iterations % 5 == 0:
            print("[%s] Working..." % time.strftime("%H:%M:%S"))

        self.gen_iterations += 1

        return lambda: self.show_sample()

    def show_sample(self):
        # get new batch of images and generate results for visualization
        _, wA, tA = self.images_A.send(14)
        _, wB, tB = self.images_B.send(14)
        self.showG(tA, tB, self.trainer_A.path, self.trainer_B.path)
        self.showG_mask(tA, tB, self.trainer_A.path_mask, self.trainer_B.path_mask)

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
        figure = figure.reshape((4,7) + figure.shape[1:])
        figure = stack_images(figure)
        figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
        return cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
        
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
        figure = figure.reshape((4,7) + figure.shape[1:])
        figure = stack_images(figure)
        figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
        return cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
