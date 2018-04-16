from .BaseTypes import TrainingDataType
from .BaseTypes import TrainingDataSample

from utils import iter_utils
import numpy as np
import cv2

'''
You can implement your own TrainingDataGenerator
'''

class TrainingDataGeneratorBase(object):
    
    #DONT OVERRIDE
    #use YourOwnTrainingDataGenerator (..., your_opt=1)
    #and then this opt will be passed in YourOwnTrainingDataGenerator.onInitialize ( your_opt )
    def __init__ (self, modelbase, trainingdatatype, batch_size=1, **kwargs):    
        if not isinstance(trainingdatatype, TrainingDataType):
            raise Exception('TrainingDataGeneratorBase() trainingdatatype is not TrainingDataType')

        self.debug = modelbase.is_debug()
        self.batch_size = 1 if self.debug else batch_size        
        self.trainingdatatype = trainingdatatype
        self.data = modelbase.get_training_data(trainingdatatype)
        self.data_counter = 0
        
        if self.debug:
            self.generator = iter_utils.ThisThreadGenerator ( self.batch_func )
        else:
            self.generator = iter_utils.SubprocessGenerator ( self.batch_func )
            
        self.onInitialize(**kwargs)
        
    #overridable
    def onInitialize(self, **kwargs):
        #your TrainingDataGenerator initialization here
        pass
        
    #overridable
    def onProcessSample(self, sample, debug):
        #process sample and return tuple of images for your model in onTrainOneEpoch
        return ( np.zeros( (64,64,4), dtype=np.float32 ), )
        
    def __iter__(self):
        return self
        
    def __next__(self):
        x = next(self.generator)       
        self.data_counter = x[0]
        return x[1]
        
    def batch_func(self):
        data_counter = self.data_counter
    
        data_len = len(self.data)
        if data_len == 0:
            raise ValueError('No training data provided.')
            
        if self.trainingdatatype >= TrainingDataType.SRC_YAW_SORTED and self.trainingdatatype <= TrainingDataType.DST_YAW_SORTED_AS_SRC_WITH_NEAREST:
            if all ( [ x == None for x in self.data] ):
             raise ValueError('Not enough training data. Gather more faces!')
        
        while True:                
            batches = None
            for n_batch in range(0, self.batch_size):      
               
                while True:
                    sample = None
                    if self.trainingdatatype >= TrainingDataType.SRC_YAW_SORTED and self.trainingdatatype <= TrainingDataType.DST_YAW_SORTED_AS_SRC_WITH_NEAREST:
                        idx = data_counter % data_len
                        if self.data[idx] != None:
                            idx_data_len = len(self.data[idx])
                            if idx_data_len > 0:
                                sample = self.data[idx][np.random.randint (0, idx_data_len)]
                    elif self.trainingdatatype == TrainingDataType.SRC or self.trainingdatatype <= TrainingDataType.DST:
                        sample = self.data[data_counter % data_len]
                    
                    data_counter += 1
                    
                    if sample is not None:                        
                        x = self.onProcessSample (sample, self.debug)
                        
                        if type(x) != tuple and type(x) != list:
                            raise Exception('TrainingDataGenerator.onProcessSample() returns NOT tuple/list')
                        x_len = len(x)
                        if batches is None:
                            batches = [ [] for _ in range(0,x_len) ]
                        for i in range(0,x_len):
                            batches[i].append ( x[i] )
                        break
            yield data_counter, [ np.array(batch) for batch in batches]
        
    def get_dict_state(self):
        return {'data_counter' : self.data_counter}

    def set_dict_state(self, state):
        self.data_counter = state['data_counter']

