# Based on the https://github.com/shaoanlu/faceswap-GAN repo (master/FaceSwap_GAN_v2_train.ipynb)

import cv2

from keras.layers import *
import keras.backend as K
from tensorflow.contrib.distributions import Beta
from keras.optimizers import Adam

K.set_learning_phase(1)

# # 3. Import VGGFace
# (Skip this part if you don't want to apply perceptual loss)

# If you got error ```_obtain_input_shape(...)``` error, this is because your keras version is older than vggface requirement. 
# 
# Modify ```_obtain_input_shape(...)``` in ```keras_vggface/models.py``` will solve the problem. The following is what worked for me:
# 
# ```python
# input_shape = _obtain_input_shape(input_shape,
#                                   default_size=224,
#                                   min_size=197,
#                                   data_format=K.image_data_format(),
#                                   include_top=include_top)
# ```

# from keras_vggface.vggface import VGGFace

# vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))

# vggface.summary()

use_perceptual_loss = False
use_lsgan = True
use_instancenorm = False
use_mixup = True
mixup_alpha = 0.2 # 0.2

batchSize = 32
lrD = 1e-4 # Discriminator learning rate
lrG = 1e-4 # Generator learning rate

# # 7. Define Inputs/Outputs Variables
# 
#     distorted: A (batch_size, 64, 64, 3) tensor, input of generator (netGA).
#     fake: (batch_size, 64, 64, 3) tensor, output of generator (netGA).
#     mask: (batch_size, 64, 64, 1) tensor, mask output of generator (netGA).
#     path: A function that takes distorted as input and outputs fake.
#     path_B: A function that takes distorted_B as input and outputs fake_B.
#     path_mask: A function that takes distorted as input and outputs mask.
#     path_abgr: A function that takes distorted as input and outputs concat([mask, fake]).
#     real: A (batch_size, 64, 64, 3) tensor, target images for generator given input distorted.

def cycle_variables(netG):
    distorted_input = netG.inputs[0]
    fake_output = netG.outputs[0]
    alpha = Lambda(lambda x: x[:,:,:, :1])(fake_output)
    rgb = Lambda(lambda x: x[:,:,:, 1:])(fake_output)
    
    masked_fake_output = alpha * rgb + (1-alpha) * distorted_input 

    fn_generate = K.function([distorted_input], [masked_fake_output])
    fn_mask = K.function([distorted_input], [concatenate([alpha, alpha, alpha])])
    fn_abgr = K.function([distorted_input], [concatenate([alpha, rgb])])
    return distorted_input, fake_output, alpha, fn_generate, fn_mask, fn_abgr

# # 8. Define Loss Function
# 
# LSGAN

if use_lsgan:
    loss_fn = lambda output, target : K.mean(K.abs(K.square(output-target)))
else:
    loss_fn = lambda output, target : -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target))

# ========== Define Perceptual Loss Model==========
if use_perceptual_loss:
    vggface.trainable = False
    out_size55 = vggface.layers[36].output
    out_size28 = vggface.layers[78].output
    out_size7 = vggface.layers[-2].output
    vggface_feat = Model(vggface.input, [out_size55, out_size28, out_size7])
    vggface_feat.trainable = False
else:
    vggface_feat = None

# ## Repeat Point
# 
# For **1 ~ 10000** iteratioons, set:
# ```python
# mixup = lam * concatenate([real, distorted]) + (1 - lam) * concatenate([fake, distorted])
# loss_G += K.mean(K.abs(fake_rgb - real))
# fake_sz224 = tf.image.resize_images(fake, [224, 224]) # or set use_perceptual_loss = False
# ```
#
# For **10000 ~ 13000** iterations, set:
# ```python
# mixup = lam * concatenate([real, distorted]) + (1 - lam) * concatenate([fake_rgb, distorted])
# loss_G += K.mean(K.abs(fake - real))
# fake_sz224 = tf.image.resize_images(fake, [224, 224]) # Ignore this line if you dont wan to use perceptual loss
# ```
# 
# For **13000 ~ 16000 or longer** iterations, set:
# ```python
# mixup = lam * concatenate([real, distorted]) + (1 - lam) * concatenate([fake_rgb, distorted])
# loss_G += K.mean(K.abs(fake - real))
# fake_sz224 = tf.image.resize_images(fake_rgb, [224, 224]) # Ignore this line if you dont wan to use perceptual loss
# ```

def define_loss(netD, real, fake_argb, distorted, vggface_feat=None):   
    alpha = Lambda(lambda x: x[:,:,:, :1])(fake_argb)
    fake_rgb = Lambda(lambda x: x[:,:,:, 1:])(fake_argb)
    fake = alpha * fake_rgb + (1-alpha) * distorted
    
    if use_mixup:
        dist = Beta(mixup_alpha, mixup_alpha)
        lam = dist.sample()
        # ==========
        mixup = lam * concatenate([real, distorted]) + (1 - lam) * concatenate([fake, distorted])
        # ==========
        output_mixup = netD(mixup)
        loss_D = loss_fn(output_mixup, lam * K.ones_like(output_mixup)) 
        output_fake = netD(concatenate([fake, distorted])) # dummy
        loss_G = .5 * loss_fn(output_mixup, (1 - lam) * K.ones_like(output_mixup))
    else:
        output_real = netD(real) # positive sample
        output_fake = netD(fake) # negative sample   
        loss_D_real = loss_fn(output_real, K.ones_like(output_real))    
        loss_D_fake = loss_fn(output_fake, K.zeros_like(output_fake))   
        loss_D = loss_D_real + loss_D_fake
        loss_G = .5 * loss_fn(output_fake, K.ones_like(output_fake))  
    # ==========  
    loss_G += K.mean(K.abs(fake_rgb - real))
    # ==========
    
    # Perceptual Loss
    if not vggface_feat is None:
        pl_params = (0.01, 0.1, 0.1)
        real_sz224 = tf.image.resize_images(real, [224, 224])
        # ==========
        fake_sz224 = tf.image.resize_images(fake, [224, 224]) 
        # ==========   
        real_feat55, real_feat28, real_feat7 = vggface_feat(real_sz224)
        fake_feat55, fake_feat28, fake_feat7  = vggface_feat(fake_sz224)    
        loss_G += pl_params[0] * K.mean(K.abs(fake_feat7 - real_feat7))
        loss_G += pl_params[1] * K.mean(K.abs(fake_feat28 - real_feat28))
        loss_G += pl_params[2] * K.mean(K.abs(fake_feat55 - real_feat55))
    
    return loss_D, loss_G

class Trainable():
    def __init__(self, netG, netD, shape):
        distorted, fake, mask, self.path, self.path_mask, self.path_abgr = cycle_variables(netG)
        real = Input(shape=shape)

        loss_D, loss_G = define_loss(netD, real, fake, distorted, vggface_feat)

        #loss_G += 1e-4 * K.mean(K.square(mask))

        weightsD = netD.trainable_weights
        weightsG = netG.trainable_weights

        # Adam(..).get_updates(...)
        training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsD,[],loss_D)
        self.netD_train = K.function([distorted, real],[loss_D], training_updates)

        training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(weightsG,[], loss_G)
        self.netG_train = K.function([distorted, real], [loss_G], training_updates)

    def trainD(self, batch):
        return self.netD_train(batch)

    def trainG(self, batch):
        return self.netG_train(batch)

    # # 11. Helper Function: face_swap()
    # This function is provided for those who don't have enough VRAM to run dlib's CNN and GAN model at the same time.
    # 
    #     INPUTS:
    #         img: A RGB face image of any size.
    #     OUPUTS:
    #         result_img: A RGB swapped face image after masking.
    #         result_mask: A single channel uint8 mask image.

    def swap_face(self, img):
        input_size = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # generator expects BGR input    
        ae_input = cv2.resize(img, (64,64))/255. * 2 - 1        
        
        result = np.squeeze(np.array([self.path_abgr([[ae_input]])]))
        result_a = result[:,:,0] * 255
        result_bgr = np.clip( (result[:,:,1:] + 1) * 255 / 2, 0, 255 )
        result_a = np.expand_dims(result_a, axis=2)
        result = (result_a/255 * result_bgr + (1 - result_a/255) * ((ae_input + 1) * 255 / 2)).astype('uint8')
        
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) 
        result = cv2.resize(result, (input_size[1],input_size[0]))
        result_a = np.expand_dims(cv2.resize(result_a, (input_size[1],input_size[0])), axis=2)
        return result, result_a
