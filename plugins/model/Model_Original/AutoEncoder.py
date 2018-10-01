# AutoEncoder base classes

from lib.utils import backup_file
from lib import Serializer
from json import JSONDecodeError

hdf = {'encoderH5': 'encoder.h5',
       'decoder_AH5': 'decoder_A.h5',
       'decoder_BH5': 'decoder_B.h5',
       'state': 'state'}

class AutoEncoder:
    def __init__(self, model_dir, gpus):
        self.model_dir = model_dir
        self.gpus = gpus

        self.encoder = self.Encoder()
        self.decoder_A = self.Decoder()
        self.decoder_B = self.Decoder()

        self.initModel()

    def load(self, swapped):
        serializer = Serializer.get_serializer('json')
        state_fn = ".".join([hdf['state'], serializer.ext]) 
        try:
            with open(str(self.model_dir / state_fn), 'rb') as fp:
                state = serializer.unmarshal(fp.read().decode('utf-8'))
                self._epoch_no = state['epoch_no']
        except IOError as e:
            print('Error loading training info:', e.strerror)
            self._epoch_no = 0
        except JSONDecodeError as e:
            print('Error loading training info:', e.msg)
            self._epoch_no = 0   
        
        (face_A,face_B) = (hdf['decoder_AH5'], hdf['decoder_BH5']) if not swapped else (hdf['decoder_BH5'], hdf['decoder_AH5'])                

        try:
            self.encoder.load_weights(str(self.model_dir / hdf['encoderH5']))
            self.decoder_A.load_weights(str(self.model_dir / face_A))
            self.decoder_B.load_weights(str(self.model_dir / face_B))
            print('loaded model weights')
            return True
        except Exception as e:
            print('Failed loading existing training data.')
            print(e)
            return False                            

    def save_weights(self):
        model_dir = str(self.model_dir)
        for model in hdf.values():
            backup_file(model_dir, model)
        self.encoder.save_weights(str(self.model_dir / hdf['encoderH5']))
        self.decoder_A.save_weights(str(self.model_dir / hdf['decoder_AH5']))
        self.decoder_B.save_weights(str(self.model_dir / hdf['decoder_BH5']))
        
        print('saved model weights')
        
        serializer = Serializer.get_serializer('json')
        state_fn = ".".join([hdf['state'], serializer.ext])
        state_dir = str(self.model_dir / state_fn)                        
        try:
            with open(state_dir, 'wb') as fp:
                state_json = serializer.marshal({
                    'epoch_no' : self.epoch_no
                     })
                fp.write(state_json.encode('utf-8'))
        except IOError as e:
            print(e.strerror)                   

    @property
    def epoch_no(self):
        "Get current training epoch number"
        return self._epoch_no
        
        
