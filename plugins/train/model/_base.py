#!/usr/bin/env python3
""" Base class for Models. ALL Models should at least inherit from this class

    When inheriting model_data should be a list of NNMeta objects.
    See the class for details.

"""
import os
import sys
from json import JSONDecodeError
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from lib import Serializer
from plugins.train._config import Config


class ModelBase():
    """ Base class that all models should inherit from """
    def __init__(self, model_dir, gpus, image_shape=None, encoder_dim=None):
        self.config = Config().config
        self.model_dir = model_dir
        self.gpus = gpus
        self.image_shape = image_shape
        self.encoder_dim = encoder_dim

        # Training information specific to the model should be placed in this
        # dict for reference by the trainer.
        self.training_opts = dict()

        # For autoencoder models, autoencoders should be placed in this dict
        self.autoencoders = dict()

        self.name = self.set_model_name()
        self.serializer = Serializer.get_serializer('json')
        self._epoch_no = self.load_state()
        self.networks = list()

        self.add_networks()
        self.initialize()

    def initialize(self):
        """ Override for Model Specific Initialization """
        raise NotImplementedError

    def add_networks(self):
        """ Override to add neural networks """
        raise NotImplementedError

    def add_network(self, network_type, side, network):
        """ Add a NNMeta object to self.models """
        filename = "{}_{}".format(self.name, network_type.lower())
        if side:
            filename += "_{}".format(side.upper())
        filename += ".h5"
        self.networks.append(NNMeta(str(self.model_dir / filename),
                                    network_type,
                                    side,
                                    network))

    def set_model_name(self):
        """ Set the model name based on the subclass """
        basename = os.path.basename(sys.modules[self.__module__].__file__)
        return os.path.splitext(basename)[0].lower()

    def compile_autoencoders(self):
        """ Compile the autoencoders """
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
        if self.gpus > 1:
            for acr in self.autoencoders.keys():
                autoencoder = multi_gpu_model(self.autoencoders[acr],
                                              self.gpus)
                self.autoencoders[acr] = autoencoder

        for autoencoder in self.autoencoders.values():
            autoencoder.compile(optimizer=optimizer,
                                loss='mean_absolute_error')

    def converter(self, swap):
        """ Converter for autoencoder models """
        if swap:
            return self.autoencoders["a"].predict
        return self.autoencoders["b"].predict

    @property
    def epoch_no(self):
        "Get current training epoch number"
        return self._epoch_no

    def load_state(self):
        """ Load epoch number from state file """
        epoch_no = 0
        try:
            with open(self.state_filename(), 'rb') as inp:
                state = self.serializer.unmarshal(inp.read().decode('utf-8'))
                epoch_no = state['epoch_no']
        except IOError as err:
            print('No existing training info found: {}'.format(err.strerror))
            epoch_no = 0
        except JSONDecodeError as err:
            epoch_no = 0
        return epoch_no

    def save_state(self):
        """ Save epoch number to state file """
        try:
            with open(self.state_filename(), 'wb') as out:
                state = {'epoch_no': self.epoch_no}
                state_json = self.serializer.marshal(state)
                out.write(state_json.encode('utf-8'))
        except IOError as err:
            print(err.strerror)

    def state_filename(self):
        """ Return full filepath for this models state file """
        filename = "{}_state.{}".format(self.name, self.serializer.ext)
        return str(self.model_dir / filename)

    def map_weights(self, swapped):
        """ Map the weights for A/B side models for swapping """
        weights_map = {"A": dict(), "B": dict()}
        side_a, side_b = ("A", "B") if not swapped else ("B", "A")
        for network in self.networks:
            if network.side == side_a:
                weights_map["A"][network.type] = network.filename
            if network.side == side_b:
                weights_map["B"][network.type] = network.filename
        return weights_map

    def load_weights(self, swapped):
        """ Load weights from the weights file """
        weights_mapping = self.map_weights(swapped)
        try:
            for network in self.networks:
                if not network.side:
                    network.load_weights()
                else:
                    network.load_weights(
                        weights_mapping[network.side][network.type])
            print('loaded model weights')
            return True
        except Exception as err:
            print('Failed loading existing training data.')
            print(err)
            return False

    def save_weights(self):
        """ Save the weights files """
        self.backup_weights()
        for network in self.networks:
            network.save_weights()
        print('saved model weights')
        self.save_state()

    def backup_weights(self):
        """ Backup the weights files by appending .bk to the end """
        for network in self.networks:
            origfile = network.filename
            backupfile = origfile + '.bk'
            if os.path.exists(backupfile):
                os.remove(backupfile)
            if os.path.exists(origfile):
                os.rename(origfile, backupfile)


class NNMeta():
    """ Class to hold a neural network and it's meta data

    filename:   The full path and filename of the weights file for
                this network.
    type:       The type of network. For networks that can be swapped
                The type should be identical for the corresponding
                A and B networks, and should be unique for every A/B pair.
                Otherwise the type should be completely unique.
    side:       A, B or None. Used to identify which networks can
                be swapped.
    network:      Define network to this.
    """

    def __init__(self, filename, network_type, side, network):
        self.filename = filename
        self.type = network_type
        self.side = side
        self.network = network

    def load_weights(self, fullpath=None):
        """ Load model weights """
        fullpath = fullpath if fullpath else self.filename
        self.network.load_weights(fullpath)

    def save_weights(self, fullpath=None):
        """ Load model weights """
        fullpath = fullpath if fullpath else self.filename
        self.network.save_weights(fullpath)
