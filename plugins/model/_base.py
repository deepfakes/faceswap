#!/usr/bin/env python3
""" Base class for Models. ALL Models should at least inherit from this class

    When inheriting model_data should be a list of ModelMeta objects.
    See the class for details.

"""
import os

from lib import Serializer
from json import JSONDecodeError


class ModelBase():
    def __init__(self, model_dir, gpus, model_data=None):
        self.model_dir = model_dir
        self.gpus = gpus
        self.models = self.set_weights_path(model_data)

        self.serializer = Serializer.get_serializer('json')
        self._epoch_no = self.load_state()

        self.initModel()

    def initModel(self):
        """ Override for Model Specific Initialization """
        raise NotImplementedError

    def get_model(self, filename, model_type, side, model):
        """ Return a ModelMeta object """
        return ModelMeta(filename, model_type, side, model)
    
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
            with open(str(self.model_dir / state_fn), 'rb') as fp:
                state = self.serializer.unmarshal(fp.read().decode('utf-8'))
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
            with open(state_dir, 'wb') as fp:
                state_json = self.serializer.marshal({
                    'epoch_no' : self.epoch_no
                     })
                fp.write(state_json.encode('utf-8'))
        except IOError as err:
            print(err.strerror)      

    def map_weights(self, swapped):
        """ Map the weights for A/B side models for swapping """
        weights_map = {"A": dict(), "B": dict()}
        side_a, side_b = "A", "B" if not swapped else "B", "A"
        for model in self.models:
            if model.side == side_a:
                weights_map["A"][model.type] = model.filename
            if model.side == side_b:
                weights_map["B"][model.type] = model.filename
        return weights_map

    def load_weights(self, swapped):
        """ Load weights from the weights file """
        weights_mapping = self.map_weights(swapped)
        try:
            for model in self.models:
                if not model.side:
                    model.load_weights()
                else:
                    model.load_weights(weights_mapping[model.side][model.type])
            print('loaded model weights')
            return True
        except Exception as e:
            print('Failed loading existing training data.')
            print(e)
            return False      

    def save_weights(self):
        self.backup_weights()
        for model in self.models:
            model.save_weights()
        print('saved model weights')
        self.save_state()     

    def backup_weights(self):
        """ Backup the weights files by appending .bk to the end """
        for model in self.models:
            origfile = model.filename
            backupfile = origfile + '.bk'
            if os.path.exists(backupfile):
                os.remove(backupfile)
            if os.path.exists(origfile):
                os.rename(origfile, backupfile)

class ModelMeta():
    """ Class to hold the model and it's meta data

    filename:   The filename of the weights file for this model as
                stored in the model_dir.
    type:       The type of model. For models that can be swapped
                The type should be identical for the corresponding 
                A and B models, and should be unique for every A/B pair.
                Otherwise the type should be completely unique.
    side:       A, B or None. Used to identify which models can
                be swapped.
    model:      Define models to this.
    """
    
    def __init__(self, filename, model_type, side, model):
        self.filename = filename
        self.type = model_type
        self.side = side
        self.model = model

    def load_weights(self, fullpath=None):
        """ Load model weights """
        fullpath = fullpath if fullpath else self.filename 
        self.model.load_weights(fullpath)
    
    def save_weights(self, fullpath=None):
        """ Load model weights """
        fullpath = fullpath if fullpath else self.filename 
        self.model.save_weights(fullpath)
