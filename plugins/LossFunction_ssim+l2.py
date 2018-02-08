from keras import losses
from keras_contrib.losses import DSSIMObjective

# SSIM = Structural Image Similarity, L2 =  Mean of Squared Error
      
class LossFunction(object):
    def __init__(self, mix_factor):
        self.mix_factor_a = mix_factor
        self.mix_factor_b = 1.0 - mix_factor    
        print("Loss Function : SSIM ({:.5f}) + L2 ({:.5f})".format(self.mix_factor_a, self.mix_factor_b))
        self.ssim=DSSIMObjective()
		
    def __call__(self, y_true, y_pred):
        return (self.mix_factor_a * self.ssim(y_true, y_pred) + self.mix_factor_b * losses.mean_squared_error(y_true, y_pred))        
        
        