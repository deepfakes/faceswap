#!/usr/bin/env python3
""" faceswap-GAN v2.2 Model
    Based on the Shanlou model: https://github.com/shaoanlu/faceswap-GAN """


from keras.models import Model as KerasModel
from keras.layers import concatenate, Conv2D, Dense, Flatten, Input, K, Lambda, Reshape
from keras.optimizers import Adam
import tensorflow as tf


# from .losses import *

from lib.train.nn_blocks import (conv_gan, conv_d_gan, self_attn_block,
                                 res_block_gan, upscale_nn, upscale_ps)

from .original import logger, Model as OriginalModel


class Model(OriginalModel):
    """
    faceswap-GAN v2.2 model

    Attributes:
        nc_g_inp: int, number of generator input channels
        nc_d_inp: int, number of discriminator input channels
        lr_g: float, learning rate of the generator
        lr_d: float, learning rate of the discriminator
    """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)
        self.use_self_attn = self.config.get("GAN_2-2", "use_self_attention")
        self.norm = self.config.get("GAN_2-2", "normalization")
        self.model_capacity = self.config.get("GAN_2-2", "model_capacity")

        self.nc_g_inp = 3
        self.nc_d_inp = 6
        self.enc_nc_out = 256 if self.model_capacity == "light" else 512
        self.lr_d = 2e-4
        self.lr_g = 1e-4
        self.variables = dict()

        resolution = self.config.get("GAN_2-2", "resolution")
        kwargs["image_shape"] = (resolution, resolution, 3)
        super().__init__(*args, **kwargs)

    def add_networks(self):
        """ Add the GAN v2.2 model networks """
        logger.debug("Adding networks")
        self.add_network("encoder", None, self.encoder())
        self.add_network("decoder", "A", self.decoder())
        self.add_network("decoder", "B", self.decoder())
        self.add_network("discriminator", "A", self.discriminator())
        self.add_network("discriminator", "B", self.discriminator())
        logger.debug("Added networks")

    def initialize(self):
        """ Initialize GAN model """
        logger.debug("Initializing model")
        inp = Input(shape=self.image_shape)  # dummy input tensor
        for network in self.networks:
            if network.type == "encoder":
                encoder = network.network
            elif network.type == "decoder" and network.side == "A":
                decoder_a = network.network
            elif network.type == "decoder" and network.side == "B":
                decoder_b = network.network
            elif network.type == "discriminator" and network.side == "A":
                self.autoencoders["da"] = network.network
            elif network.type == "discriminator" and network.side == "B":
                self.autoencoders["db"] = network.network

        self.log_summary("encoder", encoder)
        self.log_summary("decoder", decoder_a)
        self.log_summary("discriminator", self.autoencoders["da"])

        self.autoencoders["a"] = KerasModel(inp, decoder_a(encoder(inp)))
        self.autoencoders["b"] = KerasModel(inp, decoder_b(encoder(inp)))

        self.variables["a"] = self.define_variables(netG=self.autoencoders["a"])
        self.variables["b"] = self.define_variables(netG=self.autoencoders["a"])
        logger.debug("Initialized model")

    @staticmethod
    def icnr_keras(shape, dtype=None):
        """
        Custom initializer for subpix upscaling
        From https://github.com/kostyaev/ICNR
        Note: upscale factor is fixed to 2, and the base initializer is fixed to random normal.
        """
        shape = list(shape)
        scale = 2
        initializer = tf.keras.initializers.RandomNormal(0, 0.02)

        new_shape = shape[:3] + [int(shape[3] / (scale ** 2))]
        var_x = initializer(new_shape, dtype)
        var_x = tf.transpose(var_x, perm=[2, 0, 1, 3])
        var_x = tf.image.resize_nearest_neighbor(var_x, size=(shape[0] * scale, shape[1] * scale))
        var_x = tf.space_to_depth(var_x, block_size=scale)
        var_x = tf.transpose(var_x, perm=[1, 2, 0, 3])
        return var_x

    def encoder(self):
        """ Build the GAN Encoder """
        input_size = self.image_shape[0]
        coef = 2 if self.model_capacity == "light" else 1
        latent_dim = 2048 if (self.model_capacity == "light" and input_size > 64) else 1024
        activ_map_size = input_size
        use_norm = False if (self.norm == 'none') else True

        if self.model_capacity == "light":
            upscale_block = upscale_nn
            upscale_kwargs = dict()
        else:
            upscale_block = upscale_ps
            upscale_kwargs = {"initializer": self.icnr_keras}

        inp = Input(shape=(input_size, input_size, self.nc_g_inp))
        var_x = Conv2D(64 // coef,
                       kernel_size=5,
                       use_bias=False,  # use_bias should be True
                       padding="same")(inp)
        var_x = conv_gan(var_x, 128 // coef)
        var_x = conv_gan(var_x, 256 // coef, use_norm, norm=self.norm)
        var_x = self_attn_block(var_x, 256 // coef) if self.use_self_attn else var_x
        var_x = conv_gan(var_x, 512 // coef, use_norm, norm=self.norm)
        var_x = self_attn_block(var_x, 512 // coef) if self.use_self_attn else var_x
        var_x = conv_gan(var_x, 1024 // (coef**2), use_norm, norm=self.norm)

        activ_map_size = activ_map_size // 16
        while activ_map_size > 4:
            var_x = conv_gan(var_x, 1024 // (coef**2), use_norm, norm=self.norm)
            activ_map_size = activ_map_size // 2

        var_x = Dense(latent_dim)(Flatten()(var_x))
        var_x = Dense(4 * 4 * 1024 // (coef**2))(var_x)
        var_x = Reshape((4, 4, 1024 // (coef**2)))(var_x)
        out = upscale_block(var_x, 512 // coef, use_norm, norm=self.norm, **upscale_kwargs)
        return KerasModel(inputs=inp, outputs=out)

    def decoder(self):
        """ Build the GAN Decoder """
        input_size = 8
        output_size = self.image_shape[0]

        coef = 2 if self.model_capacity == "light" else 1
        upscale_block = upscale_nn if self.model_capacity == "light" else upscale_ps
        activ_map_size = input_size
        use_norm = False if self.norm == 'none' else True

        inp = Input(shape=(input_size, input_size, self.enc_nc_out))
        var_x = inp
        var_x = upscale_block(var_x, 256 // coef, use_norm, norm=self.norm)
        var_x = upscale_block(var_x, 128 // coef, use_norm, norm=self.norm)
        var_x = self_attn_block(var_x, 128 // coef) if self.use_self_attn else var_x
        var_x = upscale_block(var_x, 64 // coef, use_norm, norm=self.norm)
        var_x = res_block_gan(var_x, 64 // coef, norm=self.norm)
        if self.use_self_attn:
            var_x = self_attn_block(var_x, 64 // coef)
        else:
            var_x = conv_gan(var_x, 64 // coef, strides=1)

        outputs = []
        activ_map_size = activ_map_size * 8
        while activ_map_size < output_size:
            outputs.append(Conv2D(3, kernel_size=5, padding='same', activation="tanh")(var_x))
            var_x = upscale_block(var_x, 64 // coef, use_norm, norm=self.norm)
            var_x = conv_gan(var_x, 64 // coef, strides=1)
            activ_map_size *= 2

        alpha = Conv2D(1, kernel_size=5, padding='same', activation="sigmoid")(var_x)
        bgr = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(var_x)
        out = concatenate([alpha, bgr])
        outputs.append(out)
        return KerasModel(inp, outputs)

    def discriminator(self):
        """ Build the GAN Discriminator """
        input_size = self.image_shape[0]
        activ_map_size = input_size
        use_norm = False if self.norm == 'none' else True

        inp = Input(shape=(input_size, input_size, self.nc_d_inp))
        var_x = conv_d_gan(inp, 64, False)
        var_x = conv_d_gan(var_x, 128, use_norm, norm=self.norm)
        var_x = conv_d_gan(var_x, 256, use_norm, norm=self.norm)
        var_x = self_attn_block(var_x, 256) if self.use_self_attn else var_x

        activ_map_size = activ_map_size//8
        while activ_map_size > 8:
            var_x = conv_d_gan(var_x, 256, use_norm, norm=self.norm)
            var_x = self_attn_block(var_x, 256) if self.use_self_attn else var_x
            activ_map_size = activ_map_size // 2

        out = Conv2D(1,
                     kernel_size=4,
                     use_bias=False,  # use_bias should be True
                     padding="same")(var_x)
        return KerasModel(inputs=[inp], outputs=out)

    def define_variables(self, net_g):
        """ Define the GAN Variables """
        variables = dict()
        distorted_input = net_g.inputs[0]
        fake_output = net_g.outputs[-1]
        alpha = Lambda(lambda var_x: var_x[:, :, :, :1])(fake_output)
        bgr = Lambda(lambda var_x: var_x[:, :, :, 1:])(fake_output)

        masked_fake_output = alpha * bgr + (1-alpha) * distorted_input

        variables["fn_generate"] = K.function([distorted_input], [masked_fake_output])
        variables["fn_mask"] = K.function([distorted_input], [concatenate([alpha, alpha, alpha])])
        variables["fn_abgr"] = K.function([distorted_input], [concatenate([alpha, bgr])])
        variables["fn_bgr"] = K.function([distorted_input], [bgr])
        variables["distorted_input"] = distorted_input
        variables["fake_output"] = fake_output
        variables["alpha"] = alpha
        variables["real"] = Input(shape=self.image_shape)
        variables["mask_eyes"] = Input(shape=self.image_shape)
        return variables

    def build_train_functions(self, loss_weights=None, **loss_config):
        assert loss_weights is not None, "loss weights are not provided."
        # Adversarial loss
        loss_DA, loss_adv_GA = adversarial_loss(self.autoencoders["da"],
                                                self.variables["a"]["real"],
                                                self.variables["a"]["fake_output"],
                                                self.variables["a"]["distorted_input"],
                                                loss_config["gan_training"],
                                                **loss_weights)
        loss_DB, loss_adv_GB = adversarial_loss(self.autoencoders["db"],
                                                self.variables["b"]["real"],
                                                self.variables["b"]["fake_output"],
                                                self.variables["b"]["distorted_input"],
                                                loss_config["gan_training"],
                                                **loss_weights)

        # Reconstruction loss
        loss_recon_GA = reconstruction_loss(self.variables["a"]["real"],
                                            self.variables["a"]["fake_output"],
                                            self.variables["a"]["mask_eyes"],
                                            self.autoencoders["a"].outputs,
                                            **loss_weights)
        loss_recon_GB = reconstruction_loss(self.variables["b"]["real"],
                                            self.variables["b"]["fake_output"],
                                            self.variables["b"]["mask_eyes"],
                                            self.autoencoders["b"].outputs,
                                            **loss_weights)

        # Edge loss
        loss_edge_GA = edge_loss(self.variables["a"]["real"],
                                 self.variables["a"]["fake_output"],
                                 self.variables["a"]["mask_eyes"],
                                 **loss_weights)
        loss_edge_GB = edge_loss(self.variables["b"]["real"],
                                 self.variables["b"]["fake_output"],
                                 self.variables["b"]["mask_eyes"],
                                 **loss_weights)

        if loss_config['use_PL']:
            loss_pl_GA = perceptual_loss(self.variables["a"]["real"],
                                         self.variables["a"]["fake_output"],
                                         self.variables["a"]["distorted_input"],
                                         self.variables["a"]["mask_eyes"],
                                         self.vggface_feats, **loss_weights)
            loss_pl_GB = perceptual_loss(self.variables["b"]["real"],
                                         self.variables["b"]["fake_output"],
                                         self.variables["b"]["distorted_input"],
                                         self.variables["b"]["mask_eyes"],
                                         self.vggface_feats, **loss_weights)
        else:
            loss_pl_GA = loss_pl_GB = K.zeros(1)

        loss_GA = loss_adv_GA + loss_recon_GA + loss_edge_GA + loss_pl_GA
        loss_GB = loss_adv_GB + loss_recon_GB + loss_edge_GB + loss_pl_GB

        # The following losses are rather trivial, thus their weights are fixed.
        # Cycle consistency loss
        if loss_config['use_cyclic_loss']:
            loss_GA += 10 * cyclic_loss(self.autoencoders["a"],
                                        self.autoencoders["b"],
                                        self.variables["a"]["real"])
            loss_GB += 10 * cyclic_loss(self.autoencoders["b"],
                                        self.autoencoders["a"],
                                        self.variables["b"]["real"])

        # Alpha mask loss
        if not loss_config['use_mask_hinge_loss']:
            loss_GA += 1e-2 * K.mean(K.abs(self.variables["a"]["alpha"]))
            loss_GB += 1e-2 * K.mean(K.abs(self.variables["b"]["alpha"]))
        else:
            loss_GA += 0.1 * K.mean(
                K.maximum(0., loss_config['m_mask'] - self.variables["a"]["alpha"]))
            loss_GB += 0.1 * K.mean(
                K.maximum(0., loss_config['m_mask'] - self.variables["b"]["alpha"]))

        # Alpha mask total variation loss
        loss_GA += 0.1 * K.mean(first_order(self.variables["a"]["alpha"], axis=1))
        loss_GA += 0.1 * K.mean(first_order(self.variables["a"]["alpha"], axis=2))
        loss_GB += 0.1 * K.mean(first_order(self.variables["b"]["alpha"], axis=1))
        loss_GB += 0.1 * K.mean(first_order(self.variables["b"]["alpha"], axis=2))

        # L2 weight decay
        # https://github.com/keras-team/keras/issues/2662
        for loss_tensor in self.autoencoders["a"].losses:
            loss_GA += loss_tensor
        for loss_tensor in self.autoencoders["b"].losses:
            loss_GB += loss_tensor
        for loss_tensor in self.autoencoders["da"].losses:
            loss_DA += loss_tensor
        for loss_tensor in self.autoencoders["db"].losses:
            loss_DB += loss_tensor

        weightsDA = self.autoencoders["da"].trainable_weights
        weightsGA = self.autoencoders["a"].trainable_weights
        weightsDB = self.autoencoders["db"].trainable_weights
        weightsGB = self.autoencoders["b"].trainable_weights

        # Define training functions
        # Adam(...).get_updates(...)
        training_updates = Adam(lr=self.lr_d*loss_config['lr_factor'],
                                beta_1=0.5).get_updates(weightsDA, [], loss_DA)
        self.netDA_train = K.function([self.variables["a"]["distorted_input"],
                                       self.variables["a"]["real"]],
                                      [loss_DA],
                                      training_updates)
        training_updates = Adam(lr=self.lr_g*loss_config['lr_factor'],
                                beta_1=0.5).get_updates(weightsGA, [], loss_GA)
        self.netGA_train = K.function([self.variables["a"]["distorted_input"],
                                       self.variables["a"]["real"],
                                       self.variables["a"]["mask_eyes"]],
                                      [loss_GA,
                                       loss_adv_GA,
                                       loss_recon_GA,
                                       loss_edge_GA,
                                       loss_pl_GA],
                                      training_updates)

        training_updates = Adam(lr=self.lr_d*loss_config['lr_factor'],
                                beta_1=0.5).get_updates(weightsDB, [], loss_DB)
        self.netDB_train = K.function([self.variables["b"]["distorted_input"],
                                       self.variables["b"]["real"]],
                                      [loss_DB],
                                      training_updates)
        training_updates = Adam(lr=self.lr_g*loss_config['lr_factor'],
                                beta_1=0.5).get_updates(weightsGB, [], loss_GB)
        self.netGB_train = K.function([self.variables["b"]["distorted_input"],
                                       self.variables["b"]["real"],
                                       self.variables["b"]["mask_eyes"]],
                                      [loss_GB,
                                       loss_adv_GB,
                                       loss_recon_GB,
                                       loss_edge_GB,
                                       loss_pl_GB],
                                      training_updates)

    def build_pl_model(self, vggface_model, before_activ=False):
        # Define Perceptual Loss Model
        vggface_model.trainable = False
        if not before_activ:
            out_size112 = vggface_model.layers[1].output
            out_size55 = vggface_model.layers[36].output
            out_size28 = vggface_model.layers[78].output
            out_size7 = vggface_model.layers[-2].output
        else:
            out_size112 = vggface_model.layers[15].output  # misnamed: the output size is 55
            out_size55 = vggface_model.layers[35].output
            out_size28 = vggface_model.layers[77].output
            out_size7 = vggface_model.layers[-3].output
        self.vggface_feats = KerasModel(vggface_model.input,
                                        [out_size112, out_size55, out_size28, out_size7])
        self.vggface_feats.trainable = False

    def load_weights(self, path="./models"):
        try:
            self.encoder.load_weights(f"{path}/encoder.h5")
            self.decoder_A.load_weights(f"{path}/decoder_A.h5")
            self.decoder_B.load_weights(f"{path}/decoder_B.h5")
            self.autoencoders["da"].load_weights(f"{path}/netDA.h5")
            self.autoencoders["db"].load_weights(f"{path}/netDB.h5")
            print("Model weights files are successfully loaded.")
        except:
            print("Error occurs during loading weights files.")
            pass

    def save_weights(self, path="./models"):
        try:
            self.encoder.save_weights(f"{path}/encoder.h5")
            self.decoder_A.save_weights(f"{path}/decoder_A.h5")
            self.decoder_B.save_weights(f"{path}/decoder_B.h5")
            self.autoencoders["da"].save_weights(f"{path}/netDA.h5")
            self.autoencoders["db"].save_weights(f"{path}/netDB.h5")
            print(f"Model weights files have been saved to {path}.")
        except:
            print("Error occurs during saving weights.")
            pass

    def train_one_batch_G(self, data_A, data_B):
        if len(data_A) == 4 and len(data_B) == 4:
            _, warped_A, target_A, bm_eyes_A = data_A
            _, warped_B, target_B, bm_eyes_B = data_B
        elif len(data_A) == 3 and len(data_B) == 3:
            warped_A, target_A, bm_eyes_A = data_A
            warped_B, target_B, bm_eyes_B = data_B
        else:
            raise ValueError("Something's wrong with the input data generator.")
        errGA = self.netGA_train([warped_A, target_A, bm_eyes_A])
        errGB = self.netGB_train([warped_B, target_B, bm_eyes_B])
        return errGA, errGB

    def train_one_batch_D(self, data_A, data_B):
        if len(data_A) == 4 and len(data_B) == 4:
            _, warped_A, target_A, _ = data_A
            _, warped_B, target_B, _ = data_B
        elif len(data_A) == 3 and len(data_B) == 3:
            warped_A, target_A, _ = data_A
            warped_B, target_B, _ = data_B
        else:
            raise ValueError("Something's wrong with the input data generator.")
        errDA = self.netDA_train([warped_A, target_A])
        errDB = self.netDB_train([warped_B, target_B])
        return errDA, errDB

    def transform_a2b(self, img):
        """ Transform A to B """
        return self.variables["b"]["fn_abgr"]([[img]])

    def transform_b2a(self, img):
        """ Transform B to A """
        return self.variables["a"]["fn_abgr"]([[img]])
