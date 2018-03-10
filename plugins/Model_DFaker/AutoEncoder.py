# AutoEncoder base classes

encoderH5 = 'DF_encoder.h5'
decoder_AH5 = 'DF_decoder_A.h5'
decoder_BH5 = 'DF_decoder_B.h5'

class AutoEncoder:
    def __init__(self, model_dir):
        self.model_dir = model_dir

        self.encoder = Encoder()
        self.decoder_A = Decoder('MA')
        self.decoder_B = Decoder('MB')

        self.initModel()

    def load(self, swapped):
        (face_A,face_B) = (decoder_AH5, decoder_BH5) if not swapped else (decoder_BH5, decoder_AH5)

        try:
            self.encoder.load_weights(str(self.model_dir / encoderH5))
            self.decoder_A.load_weights(str(self.model_dir / face_A))
            self.decoder_B.load_weights(str(self.model_dir / face_B))
            print('loaded model weights')
            return True
        except Exception as e:
            print('Failed loading existing training data.')
            print(e)
            return False

    def save_weights(self):
        self.encoder.save_weights(str(self.model_dir / encoderH5))
        self.decoder_A.save_weights(str(self.model_dir / decoder_AH5))
        self.decoder_B.save_weights(str(self.model_dir / decoder_BH5))
        print('saved model weights')
