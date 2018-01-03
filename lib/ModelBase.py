
class ModelBase:
    def __init__(self, model_dir):

        self.model_dir = model_dir

        self.encoder = self.Encoder()
        self.decoder_A = self.Decoder()
        self.decoder_B = self.Decoder()

        self.initModel()

    def load(self, swapped):
        (face_A,face_B) = ('/decoder_A.h5', '/decoder_B.h5') if not swapped else ('/decoder_B.h5', '/decoder_A.h5')

        try:
            self.encoder.load_weights(self.model_dir + '/encoder.h5')
            self.decoder_A.load_weights(self.model_dir + face_A)
            self.decoder_B.load_weights(self.model_dir + face_B)
            print('loaded model weights')
        except Exception as e:
            print('Failed loading existing training data.')
            print(e)

    def save_weights(self):
        self.encoder.save_weights(self.model_dir + '/encoder.h5')
        self.decoder_A.save_weights(self.model_dir + '/decoder_A.h5')
        self.decoder_B.save_weights(self.model_dir + '/decoder_B.h5')
        print('saved model weights')