#!/usr/bin/env python3
""" Base class for Models. ALL Models should at least inherit from this class

    When inheriting model_data should be a list of NNMeta objects.
    See the class for details.

"""
import os
from json import JSONDecodeError

from lib import Serializer


class ModelBase():
    """ Base class that all models should inherit from """
    def __init__(self, model_dir, gpus, image_shape=None, encoder_dim=None):
        self.model_dir = model_dir
        self.gpus = gpus
        self.image_shape = image_shape
        self.encoder_dim = encoder_dim
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

    def add_network(self, filename, network_type, side, network):
        """ Add a NNMeta object to self.models """
        self.networks.append(NNMeta(self.model_dir,
                                    filename,
                                    network_type,
                                    side,
                                    network))

    @property
    def epoch_no(self):
        "Get current training epoch number"
        return self._epoch_no

    def set_weights_path(self, model_data):
        """ Set the model information into a dict for future use """
        for model in model_data:
            model.filename = str(self.model_dir / model.filename)
        return model_data

    def load_state(self):
        """ Load epoch number from state file """
        epoch_no = 0
        state_fn = ".".join(["state", self.serializer.ext])
        try:
            with open(str(self.model_dir / state_fn), 'rb') as inp:
                state = self.serializer.unmarshal(inp.read().decode('utf-8'))
                epoch_no = state['epoch_no']
        except IOError as err:
            print('Error loading training info:', err.strerror)
            epoch_no = 0
        except JSONDecodeError as err:
            epoch_no = 0
        return epoch_no

    def save_state(self):
        """ Save epoch number to state file """
        state_fn = ".".join(["state", self.serializer.ext])
        state_dir = str(self.model_dir / state_fn)
        try:
            with open(state_dir, 'wb') as out:
                state = {'epoch_no': self.epoch_no}
                state_json = self.serializer.marshal(state)
                out.write(state_json.encode('utf-8'))
        except IOError as err:
            print(err.strerror)

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

    model_dir:  The folder containing the weights for this model
    filename:   The filename of the weights file for this network as
                stored in the model_dir.
    type:       The type of network. For networks that can be swapped
                The type should be identical for the corresponding
                A and B networks, and should be unique for every A/B pair.
                Otherwise the type should be completely unique.
    side:       A, B or None. Used to identify which networks can
                be swapped.
    network:      Define network to this.
    """

    def __init__(self, model_dir, filename, network_type, side, network):
        self.filename = self.set_fullpath(model_dir, filename)
        self.type = network_type
        self.side = side
        self.network = network

    @staticmethod
    def set_fullpath(model_dir, filename):
        """ Set the full path to the weights file """
        fullpath = str(model_dir / filename)
        if not os.path.exists(fullpath):
            raise ValueError("Model data does not exist "
                             "at {}".format(fullpath))
        return fullpath

    def load_weights(self, fullpath=None):
        """ Load model weights """
        fullpath = fullpath if fullpath else self.filename
        self.network.load_weights(fullpath)

    def save_weights(self, fullpath=None):
        """ Load model weights """
        fullpath = fullpath if fullpath else self.filename
        self.network.save_weights(fullpath)
