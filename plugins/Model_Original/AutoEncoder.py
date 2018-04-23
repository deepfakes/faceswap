# AutoEncoder base classes

from lib.utils import backup_file

hdf = {'encoderH5': 'encoder.h5',
       'decoder_AH5': 'decoder_A.h5',
       'decoder_BH5': 'decoder_B.h5'}

class AutoEncoder:
    def __init__(self, model_dir, gpus):
        self.model_dir = model_dir
        self.gpus = gpus

        self.encoder = self.Encoder()
        self.decoder_A = self.Decoder()
        self.decoder_B = self.Decoder()

        self.initModel()

    def load(self, swapped):
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
