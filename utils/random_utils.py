import numpy as np

def random_normal( size=(1,), trunc_val = 2.5 ):
    len = np.array(size).prod()
    result = np.empty ( (len,) , dtype=np.float32)
    
    for i in range (len):
        while True:
            x = np.random.normal()
            if x >= -trunc_val and x <= trunc_val:
                break
        result[i] = (x / trunc_val)
        
    return result.reshape ( size )