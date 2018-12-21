# AutoEncoder base classes
import logging

from lib.utils import backup_file

hdf = {'encoderH5': 'lowmem_encoder.h5',
       'decoder_AH5': 'lowmem_decoder_A.h5',
       'decoder_BH5': 'lowmem_decoder_B.h5'}

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

#Part of Filename migration, should be remopved some reasonable time after first added
import os.path
old_encoderH5 = 'encoder.h5'
old_decoder_AH5 = 'decoder_A.h5'
old_decoder_BH5 = 'decoder_B.h5'
#End filename migration

class AutoEncoder:
    def __init__(self, model_dir, gpus):
        self.model_dir = model_dir
        self.gpus = gpus

        self.encoder = self.Encoder()
        self.decoder_A = self.Decoder()
        self.decoder_B = self.Decoder()

        self.initModel()

    def load(self, swapped):
        (face_A,face_B) = (hdf['decoder_AH5'], hdf['decoder_BH5']) if not swapped else (hdf['decoder_BH5'], hdf['decoder_AH5'])

        try:
            #Part of Filename migration, should be remopved some reasonable time after first added
            if os.path.isfile(str(self.model_dir / old_encoderH5)):
                logger.info('Migrating to new filenames:')
                if os.path.isfile(str(self.model_dir / hdf['encoderH5'])) is not True:
                    os.rename(str(self.model_dir / old_decoder_AH5), str(self.model_dir / hdf['decoder_AH5']))
                    os.rename(str(self.model_dir / old_decoder_BH5), str(self.model_dir / hdf['decoder_BH5']))
                    os.rename(str(self.model_dir / old_encoderH5), str(self.model_dir / hdf['encoderH5']))
                    logger.info('Complete')
                else:
                    logger.warning('Failed due to existing files in folder.  Loading already migrated files')
            #End filename migration
            self.encoder.load_weights(str(self.model_dir / hdf['encoderH5']))
            self.decoder_A.load_weights(str(self.model_dir / face_A))
            self.decoder_B.load_weights(str(self.model_dir / face_B))
            logger.info('loaded model weights')
            return True
        except Exception as e:
            logger.warning('Failed loading existing training data. Starting a fresh model: %s', self.model_dir)
            return False

    def save_weights(self):
        model_dir = str(self.model_dir)
        for model in hdf.values():
            backup_file(model_dir, model)
        self.encoder.save_weights(str(self.model_dir / hdf['encoderH5']))
        self.decoder_A.save_weights(str(self.model_dir / hdf['decoder_AH5']))
        self.decoder_B.save_weights(str(self.model_dir / hdf['decoder_BH5']))
        logger.info('saved model weights')
