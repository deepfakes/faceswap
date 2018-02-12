from keras import losses

# L1 loss function = Mean of Squared Error


class LossFunction(object):
    def __init__(self, mix_factor):
        print('Loss Function : L2')
    
    def __call__(self, y_true, y_pred):
        return losses.mean_squared_error(y_true, y_pred)