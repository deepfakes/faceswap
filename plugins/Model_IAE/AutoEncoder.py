# Improved-AutoEncoder base classes


encoderH5 = 'encoder.h5'
decoderH5 = 'decoder.h5'
inter_AH5 = 'inter_A.h5'
inter_BH5 = 'inter_B.h5'
inter_bothH5 = 'inter_both.h5'


class AutoEncoder:
    def __init__(self, model_dir):
        self.model_dir = model_dir

        self.encoder = self.Encoder()
        self.decoder = self.Decoder()
        self.inter_A = self.Intermidiate()
        self.inter_B = self.Intermidiate()
        self.inter_both = self.Intermidiate()

        self.initModel()

    def load(self, swapped):
        (face_A,face_B) = (inter_AH5, inter_BH5) if not swapped else (inter_AH5, inter_BH5)

        try:
            self.encoder.load_weights(str(self.model_dir / encoderH5))
            self.decoder.load_weights(str(self.model_dir / decoderH5))
            self.inter_both.load_weights(str(self.model_dir / inter_bothH5))
            self.inter_A.load_weights(str(self.model_dir / face_A))
            self.inter_B.load_weights(str(self.model_dir / face_B))
            print('loaded model weights')
            return True
        except Exception as e:
            print('Failed loading existing training data.')
            print(e)
            return False

    def save_weights(self):
        self.encoder.save_weights(str(self.model_dir / encoderH5))
        self.decoder.save_weights(str(self.model_dir / decoderH5))
        self.inter_both.save_weights(str(self.model_dir / inter_bothH5))
        self.inter_A.save_weights(str(self.model_dir / inter_AH5))
        self.inter_B.save_weights(str(self.model_dir / inter_BH5))
        print('saved model weights')

