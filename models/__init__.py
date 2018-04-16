from .BaseTypes import TrainingDataType
from .BaseTypes import TrainingDataSample

from .ModelBase import ModelBase
from .ConverterBase import ConverterBase
from .ConverterMasked import ConverterMasked
from .TrainingDataGeneratorBase import TrainingDataGeneratorBase
from .HalfFaceTrainingDataGenerator import HalfFaceTrainingDataGenerator
from .FullFaceTrainingDataGenerator import FullFaceTrainingDataGenerator

def import_model(name):
    module = __import__('Model_'+name, globals(), locals(), [], 1)
    return getattr(module, 'Model')