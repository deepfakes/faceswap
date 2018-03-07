# AutoEncoder base classes

encoderH5 = 'encoder.h5'
decoder_BH5 = 'decoder_B_retrained.h5'

class AutoEncoder:
    def __init__(self, model_dir, gpus):
        self.model_dir = model_dir

        self.encoder = self.Encoder()
        self.decoder_B = self.Decoder()

        self.initModel()

    def load(self, swapped):

        try:
            self.encoder.load_weights(str(self.model_dir / encoderH5))
            self.decoder_B.load_weights(str(self.model_dir / decoder_BH5))
            print('loaded model weights')
            return True
        except Exception as e:
            print('Failed loading existing training data.')
            print(e)
            return False

    def save_weights(self):
        self.decoder_B.save_weights(str(self.model_dir / decoder_BH5))
        print('saved model weights')
