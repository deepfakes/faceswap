#!/usr/bin/env python3
""" faceswap-GAN v2.2 Model
    Based on the Shanlou model: https://github.com/shaoanlu/faceswap-GAN """


from keras.models import Model as KerasModel
from keras.layers import concatenate, Conv2D, Dense, Flatten, Input, K, Lambda, Reshape
from keras.optimizers import Adam
from keras_vggface.vggface import VGGFace

from lib.model.initializers import icnr_keras
from lib.model.losses import (adversarial_loss, cyclic_loss, edge_loss, first_order,
                              perceptual_loss, reconstruction_loss)
from lib.model.nn_blocks import (conv_gan, conv_d_gan, self_attn_block,
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

        self.num_chans_g_inp = 3
        self.num_chans_d_inp = 6
        self.enc_num_chans_out = 256 if self.model_capacity == "light" else 512
        self.loss_funcs = dict()

        resolution = self.config.get("GAN_2-2", "resolution")
        kwargs["image_shape"] = (resolution, resolution, 3)
        kwargs["trainer"] = "gan_v2_2"
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

        ae_a = KerasModel(
            inp,
            self.networks["decoder_a"].network(self.networks["encoder"].network(inp)))
        ae_b = KerasModel(
            inp,
            self.networks["decoder_b"].network(self.networks["encoder"].network(inp)))
        self.add_predictors(ae_a, ae_b)

        self.log_summary("encoder", self.networks["encoder"].network)
        self.log_summary("decoder", self.networks["decoder_a"].network)
        self.log_summary("discriminator", self.networks["discriminator_a"].network)
        self.convert_multi_gpu()
        self.build_loss_functions()
        logger.debug("Initialized model")

    def compile_predictors(self):
        """ Predictors are not compiled for GAN """
        pass

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
            upscale_kwargs = {"initializer": icnr_keras}

        inp = Input(shape=(input_size, input_size, self.num_chans_g_inp))
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

        inp = Input(shape=(input_size, input_size, self.enc_num_chans_out))
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

        inp = Input(shape=(input_size, input_size, self.num_chans_d_inp))
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

    def build_loss_functions(self):
        """ Build the loss functions """
        loss_weights = {"w_D": self.config.get("GAN_2-2", "w_D"),
                        "w_recon": self.config.get("GAN_2-2", "w_recon"),
                        "w_edge": self.config.get("GAN_2-2", "w_edge"),
                        "w_eyes": self.config.get("GAN_2-2", "w_eyes"),
                        "w_pl": self.config.get("GAN_2-2", "w_pl")}
        variables = {"a": self.define_variables(net_g=self.predictors["a"]),
                     "b": self.define_variables(net_g=self.predictors["b"])}

        loss = self.build_standard_loss(loss_weights, variables)
        self.build_perceptual_loss(loss_weights, loss, variables)
        self.build_optional_loss(loss, variables)
        self.build_train_functions(loss, variables)
        return loss

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

    def build_standard_loss(self, loss_weights, variables):
        """ Build the standard loss functions """
        loss = dict()
        # Adversarial loss
        loss["da"], loss["adv_ga"] = adversarial_loss(self.networks["discriminator_a"],
                                                      variables["a"]["real"],
                                                      variables["a"]["fake_output"],
                                                      variables["a"]["distorted_input"],
                                                      self.config.get("GAN_2-2", "gan_training"),
                                                      **loss_weights)
        loss["db"], loss["adv_gb"] = adversarial_loss(self.networks["discriminator_b"],
                                                      variables["b"]["real"],
                                                      variables["b"]["fake_output"],
                                                      variables["b"]["distorted_input"],
                                                      self.config.get("GAN_2-2", "gan_training"),
                                                      **loss_weights)
        # Reconstruction loss
        loss["recon_ga"] = reconstruction_loss(variables["a"]["real"],
                                               variables["a"]["fake_output"],
                                               variables["a"]["mask_eyes"],
                                               self.predictors["a"].outputs,
                                               **loss_weights)
        loss["recon_gb"] = reconstruction_loss(variables["b"]["real"],
                                               variables["b"]["fake_output"],
                                               variables["b"]["mask_eyes"],
                                               self.predictors["b"].outputs,
                                               **loss_weights)
        # Edge loss
        loss["edge_ga"] = edge_loss(variables["a"]["real"],
                                    variables["a"]["fake_output"],
                                    variables["a"]["mask_eyes"],
                                    **loss_weights)
        loss["edge_gb"] = edge_loss(variables["b"]["real"],
                                    variables["b"]["fake_output"],
                                    variables["b"]["mask_eyes"],
                                    **loss_weights)

        loss["ga"] = loss["adv_ga"] + loss["recon_ga"] + loss["edge_ga"]
        loss["gb"] = loss["adv_gb"] + loss["recon_gb"] + loss["edge_gb"]
        return loss

    def build_perceptual_loss(self, loss_weights, loss, variables):
        """ Build the perceptual loss function """
        if self.config.get("GAN_2-2", "use_pl"):
            # VGGFace ResNet50
            vggface = VGGFace(include_top=False,
                              model='resnet50',
                              input_shape=(224, 224, 3))
            self.build_pl_model(vggface, variables)
            loss["pl_ga"] = perceptual_loss(variables["a"]["real"],
                                            variables["a"]["fake_output"],
                                            variables["a"]["distorted_input"],
                                            variables["a"]["mask_eyes"],
                                            variables["vggface_feats"],
                                            **loss_weights)
            loss["pl_gb"] = perceptual_loss(variables["b"]["real"],
                                            variables["b"]["fake_output"],
                                            variables["b"]["distorted_input"],
                                            variables["b"]["mask_eyes"],
                                            variables["vggface_feats"],
                                            **loss_weights)
        else:
            loss["pl_ga"] = loss["pl_gb"] = K.zeros(1)

        loss["ga"] += loss["pl_ga"]
        loss["gb"] += loss["pl_gb"]

    def build_pl_model(self, vggface_model, variables):
        """ Define Perceptual Loss Model """
        before_activ = self.config.get("GAN_2-2", "pl_before_activ")
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
        variables["vggface_feats"] = KerasModel(vggface_model.input,
                                                [out_size112, out_size55, out_size28, out_size7])
        variables["vggface_feats"].trainable = False

    def build_optional_loss(self, loss, variables):
        """ Add the optional loss functions

            The following losses are rather trivial, thus their weights are fixed. """
        # Cycle consistency loss
        if self.config.get("GAN_2-2", "use_cyclic_loss"):
            loss["ga"] += 10 * cyclic_loss(self.predictors["a"],
                                           self.predictors["b"],
                                           variables["a"]["real"])
            loss["gb"] += 10 * cyclic_loss(self.predictors["b"],
                                           self.predictors["a"],
                                           variables["b"]["real"])
        # Alpha mask loss
        if self.config.get("GAN_2-2", "use_mask_hinge_loss"):
            m_mask = self.config.get("GAN_2-2", "m_mask")
            loss["ga"] += 0.1 * K.mean(K.maximum(0., m_mask - variables["a"]["alpha"]))
            loss["gb"] += 0.1 * K.mean(K.maximum(0., m_mask - variables["b"]["alpha"]))
        else:
            loss["ga"] += 1e-2 * K.mean(K.abs(variables["a"]["alpha"]))
            loss["gb"] += 1e-2 * K.mean(K.abs(variables["b"]["alpha"]))
        # Alpha mask total variation loss
        loss["ga"] += 0.1 * K.mean(first_order(variables["a"]["alpha"], axis=1))
        loss["ga"] += 0.1 * K.mean(first_order(variables["a"]["alpha"], axis=2))
        loss["gb"] += 0.1 * K.mean(first_order(variables["b"]["alpha"], axis=1))
        loss["gb"] += 0.1 * K.mean(first_order(variables["b"]["alpha"], axis=2))
        # L2 weight decay
        # https://github.com/keras-team/keras/issues/2662
        for loss_tensor in self.predictors["a"].losses:
            loss["ga"] += loss_tensor
        for loss_tensor in self.predictors["b"].losses:
            loss["gb"] += loss_tensor
        for loss_tensor in self.networks["discriminator_a"].network.losses:
            loss["da"] += loss_tensor
        for loss_tensor in self.networks["discriminator_b"].network.losses:
            loss["db"] += loss_tensor

    def build_train_functions(self, loss, variables):
        """ Define training functions """
        # Adam(...).get_updates(...)
        weights = self.get_trainable_weights()
        optimizers = self.build_optimizers()
        training_updates = optimizers["discriminator"].get_updates(weights["da"], [], loss["da"])
        self.loss_funcs["da"] = K.function([variables["a"]["distorted_input"],
                                            variables["a"]["real"]],
                                           [loss["da"]],
                                           training_updates)
        training_updates = optimizers["generator"].get_updates(weights["ga"], [], loss["ga"])
        self.loss_funcs["ga"] = K.function([variables["a"]["distorted_input"],
                                            variables["a"]["real"],
                                            variables["a"]["mask_eyes"]],
                                           [loss["ga"],
                                            loss["adv_ga"],
                                            loss["recon_ga"],
                                            loss["edge_ga"],
                                            loss["pl_ga"]],
                                           training_updates)

        training_updates = optimizers["discriminator"].get_updates(weights["db"], [], loss["db"])
        self.loss_funcs["db"] = K.function([variables["b"]["distorted_input"],
                                            variables["b"]["real"]],
                                           [loss["db"]],
                                           training_updates)
        training_updates = optimizers["generator"].get_updates(weights["gb"], [], loss["gb"])
        self.loss_funcs["gb"] = K.function([variables["b"]["distorted_input"],
                                            variables["b"]["real"],
                                            variables["b"]["mask_eyes"]],
                                           [loss["gb"],
                                            loss["adv_gb"],
                                            loss["recon_gb"],
                                            loss["edge_gb"],
                                            loss["pl_gb"]],
                                           training_updates)

    def get_trainable_weights(self):
        """ Obtain the trainable weights """
        weights = dict()
        weights["da"] = self.networks["discriminator_a"].network.trainable_weights
        weights["ga"] = self.predictors["a"].trainable_weights
        weights["db"] = self.networks["discriminator_b"].network.trainable_weights
        weights["gb"] = self.predictors["b"].trainable_weights
        return weights

    def build_optimizers(self):
        """ Build the optimizers """
        optimizers = dict()
        lr_factor = self.config.get("GAN_2-2", "lr_factor")
        learning_rate_d = 2e-4 * lr_factor
        learning_rate_g = 1e-4 * lr_factor
        optimizers["discriminator"] = Adam(lr=learning_rate_d, beta_1=0.5)
        optimizers["generator"] = Adam(lr=learning_rate_g, beta_1=0.5)
        return optimizers

    def train_one_batch_g(self, data_a, data_b):
        """ Train one generator batch """
        if len(data_a) == 4 and len(data_b) == 4:
            _, warped_a, target_a, bm_eyes_a = data_a
            _, warped_b, target_b, bm_eyes_b = data_b
        elif len(data_a) == 3 and len(data_b) == 3:
            warped_a, target_a, bm_eyes_a = data_a
            warped_b, target_b, bm_eyes_b = data_b
        else:
            raise ValueError("Something's wrong with the input data generator.")
        err_ga = self.loss_funcs["ga"]([warped_a, target_a, bm_eyes_a])
        err_gb = self.loss_funcs["gb"]([warped_b, target_b, bm_eyes_b])
        return err_ga, err_gb

    def train_one_batch_d(self, data_a, data_b):
        """ Train one discriminator batch """
        if len(data_a) == 4 and len(data_b) == 4:
            _, warped_a, target_a, _ = data_a
            _, warped_b, target_b, _ = data_b
        elif len(data_a) == 3 and len(data_b) == 3:
            warped_a, target_a, _ = data_a
            warped_b, target_b, _ = data_b
        else:
            raise ValueError("Something's wrong with the input data generator.")
        err_da = self.loss_funcs["da"]([warped_a, target_a])
        err_db = self.loss_funcs["db"]([warped_b, target_b])
        return err_da, err_db

    def transform_a2b(self, img):
        """ Transform A to B """
        return self.variables["b"]["fn_abgr"]([[img]])

    def transform_b2a(self, img):
        """ Transform B to A """
        return self.variables["a"]["fn_abgr"]([[img]])
