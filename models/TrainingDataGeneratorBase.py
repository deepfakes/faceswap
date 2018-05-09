from .BaseTypes import TrainingDataType
from .BaseTypes import TrainingDataSample

from utils import iter_utils
import numpy as np
import cv2
import random
import traceback

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
        return x
        
    def batch_func(self):
        data_len = len(self.data)
        if data_len == 0:
            raise ValueError('No training data provided.')
            
        if (self.trainingdatatype == TrainingDataType.SRC or self.trainingdatatype == TrainingDataType.SRC_WITH_NEAREST) and data_len > 1500:
            print ("Warning, your src faceset contains more than 1500 faces. This can make worse result. Reduce it by sort --by hist-dissim.")
        
        if self.trainingdatatype >= TrainingDataType.SRC_YAW_SORTED and self.trainingdatatype <= TrainingDataType.SRC_YAW_SORTED_AS_DST_WITH_NEAREST:
            if all ( [ x == None for x in self.data] ):
                raise ValueError('Not enough training data. Gather more faces!')
             
        if self.trainingdatatype >= TrainingDataType.SRC and self.trainingdatatype <= TrainingDataType.DST_ONLY_1:
            shuffle_idxs = []          
        elif self.trainingdatatype >= TrainingDataType.SRC_YAW_SORTED and self.trainingdatatype <= TrainingDataType.SRC_YAW_SORTED_AS_DST_WITH_NEAREST:
            shuffle_idxs = []            
            shuffle_idxs_2D = [[]]*data_len
            
        while True:                
            
            batches = None
            for n_batch in range(0, self.batch_size):
                while True:
                    sample = None
                                
                    if self.trainingdatatype >= TrainingDataType.SRC and self.trainingdatatype <= TrainingDataType.DST_ONLY_1:
                        if len(shuffle_idxs) == 0:
                            shuffle_idxs = [ i for i in range(0, data_len) ]
                            random.shuffle(shuffle_idxs)                            
                        idx = shuffle_idxs.pop()
                        sample = self.data[ idx ]
                    elif self.trainingdatatype >= TrainingDataType.SRC_YAW_SORTED and self.trainingdatatype <= TrainingDataType.SRC_YAW_SORTED_AS_DST_WITH_NEAREST:
                        if len(shuffle_idxs) == 0:
                            shuffle_idxs = [ i for i in range(0, data_len) ]
                            random.shuffle(shuffle_idxs)
                        
                        idx = shuffle_idxs.pop()                        
                        if self.data[idx] != None:
                            if len(shuffle_idxs_2D[idx]) == 0:
                                shuffle_idxs_2D[idx] = [ i for i in range(0, len(self.data[idx])) ]
                                random.shuffle(shuffle_idxs_2D[idx])
                                
                            idx2 = shuffle_idxs_2D[idx].pop()                            
                            sample = self.data[idx][idx2]
                            
                    if sample is not None:          
                        try:
                            x = self.onProcessSample (sample, self.debug)
                        except:
                            raise Exception ("Exception occured in sample %s. Error: %s" % (sample.filename, traceback.format_exc() ) )
                        
                        if type(x) != tuple and type(x) != list:
                            raise Exception('TrainingDataGenerator.onProcessSample() returns NOT tuple/list')
                            
                        x_len = len(x)
                        if batches is None:
                            batches = [ [] for _ in range(0,x_len) ]
                            
                        for i in range(0,x_len):
                            batches[i].append ( x[i] )
                            
                        break
                        
            yield [ np.array(batch) for batch in batches]
        
    def get_dict_state(self):
        return {}

    def set_dict_state(self, state):
        pass

