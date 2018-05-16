import traceback
from pathlib import Path
from utils import Path_utils
import cv2
from tqdm import tqdm
from utils.AlignedPNG import AlignedPNG
from utils import image_utils
import shutil
import numpy as np
import time
import multiprocessing
    
class model_process_predictor(object):
    def __init__(self, sq, cq, lock):
        self.sq = sq
        self.cq = cq
        self.lock = lock
        
    def __call__(self, face):
        self.lock.acquire()
        
        self.sq.put ( {'op': 'predict', 'face' : face} )
        while True:
            if not self.cq.empty():
                obj = self.cq.get()
                obj_op = obj['op']
                if obj_op == 'predict_result':
                    self.lock.release()
                    return obj['result']
            time.sleep(0.005)
        
def model_process(model_name, model_dir, in_options, sq, cq):
    try:    
        model_path = Path(model_dir)
        
        import models 
        model = models.import_model(model_name)(model_path, **in_options)
        converter = model.get_converter(**in_options)
        converter.dummy_predict()
        
        cq.put ( {'op':'init', 'converter' : converter.copy_and_set_predictor( None ) } )

        closing = False
        while not closing:
            while not sq.empty():
                obj = sq.get()
                obj_op = obj['op']
                if obj_op == 'predict':
                    result = converter.predictor ( obj['face'] )
                    cq.put ( {'op':'predict_result', 'result':result} )
                elif obj_op == 'close':                    
                    closing = True
                    break
            time.sleep(0.005)        
                    
        model.finalize()
        
    except Exception as e:
        print ( 'Error: %s' % (str(e)))
        traceback.print_exc()
            
from utils.SubprocessorBase import SubprocessorBase
class ConvertSubprocessor(SubprocessorBase):

    #override
    def __init__(self, converter, input_path_image_paths, output_path, alignments, debug): 
        super().__init__('Converter')    
        self.converter = converter
        self.input_path_image_paths = input_path_image_paths
        self.output_path = output_path
        self.alignments = alignments
        self.debug = debug
        
        self.input_data = self.input_path_image_paths
        self.files_processed = 0
        self.faces_processed = 0
        
    #override
    def process_info_generator(self):
        r = [0] if self.debug else range(multiprocessing.cpu_count())
        for i in r:
            yield 'CPU%d' % (i), {}, {'device_idx': i,
                                      'device_name': 'CPU%d' % (i), 
                                      'converter' : self.converter, 
                                      'output_dir' : str(self.output_path), 
                                      'alignments' : self.alignments,
                                      'debug': self.debug }
     
    #override
    def get_no_process_started_message(self):
        return 'Unable to start CPU processes.'
        
    #override
    def onHostGetProgressBarDesc(self):
        return "Converting"
        
    #override
    def onHostGetProgressBarLen(self):
        return len (self.input_data)
        
    #override
    def onHostGetData(self):
        if len (self.input_data) > 0:
            return self.input_data.pop(0)            
        return None
    
    #override
    def onHostDataReturn (self, data):
        self.input_data.insert(0, data)   
        
    #override
    def onClientInitialize(self, client_dict):
        print ('Running on %s.' % (client_dict['device_name']) )
        self.device_idx  = client_dict['device_idx']
        self.device_name = client_dict['device_name']
        self.converter   = client_dict['converter']
        self.output_path = Path(client_dict['output_dir']) if 'output_dir' in client_dict.keys() else None        
        self.alignments  = client_dict['alignments']
        self.debug       = client_dict['debug']
        return None

    #override
    def onClientFinalize(self):
        pass
        
    #override
    def onClientProcessData(self, data):
        filename_path = Path(data)
        output_filename_path = self.output_path / filename_path.name
            
        files_processed = 1
        faces_processed = 0
        if filename_path.stem not in self.alignments.keys():                        
            if not self.debug:
                print ( 'no faces found for %s, copying without faces' % (filename_path.name) )                
                shutil.copy ( str(filename_path), str(output_filename_path) )                
        else:                    
            image = (cv2.imread(str(filename_path)) / 255.0).astype('float32')
            faces = self.alignments[filename_path.stem]
            for image_landmarks in faces:                
                image = self.converter.convert(image, image_landmarks, self.debug)     
                if self.debug:
                    for img in image:
                        cv2.imshow ('Debug convert', img )
                        cv2.waitKey(0)
            if not self.debug:
                cv2.imwrite (str(output_filename_path), (image*255).astype(np.uint8) )
            faces_processed = len(faces)
            
        return (files_processed, faces_processed)
        
    #override
    def onHostResult (self, data, result):
        self.files_processed += result[0]
        self.faces_processed += result[1]    
        return 1
             
    #override
    def get_start_return(self):
        return self.files_processed, self.faces_processed
        
def main (input_dir, output_dir, aligned_dir, model_dir, model_name, **in_options):
    print ("Running converter.\r\n")
    
    debug = in_options['debug']
    
    try:
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        aligned_path = Path(aligned_dir)
        model_path = Path(model_dir)
        
        if not input_path.exists():
            print('Input directory not found. Please ensure it exists.')
            return

        if output_path.exists():
            for filename in Path_utils.get_image_paths(output_path):
                Path(filename).unlink()
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            
        if not aligned_path.exists():
            print('Aligned directory not found. Please ensure it exists.')
            return
            
        if not model_path.exists():
            print('Model directory not found. Please ensure it exists.')
            return
            
        aligned_path_image_paths = Path_utils.get_image_paths(aligned_path)
        
        alignments = {}
        for filename in tqdm(aligned_path_image_paths, desc= "Collecting alignments" ):
            a_png = AlignedPNG.load( str(filename) )
            if a_png is None:
                print ( "%s - no embedded data found." % (filename) )
                continue
            d = a_png.getFaceswapDictData()
            if d is None or d['source_filename'] is None or d['source_rect'] is None or d['source_landmarks'] is None:
                print ( "%s - no embedded data found." % (filename) )
                continue
            
            source_filename_stem = Path(d['source_filename']).stem
            if source_filename_stem not in alignments.keys():
                alignments[ source_filename_stem ] = []

            alignments[ source_filename_stem ].append ( np.array(d['source_landmarks']) )
            
        model_sq = multiprocessing.Queue()
        model_cq = multiprocessing.Queue()
        model_lock = multiprocessing.Lock()
        
        model_p = multiprocessing.Process(target=model_process, args=(model_name, model_dir, in_options, model_sq, model_cq))
        model_p.start()
        
        while True:
            if not model_cq.empty():
                obj = model_cq.get()
                obj_op = obj['op']
                if obj_op == 'init':
                    converter = obj['converter']
                    break

        files_processed, faces_processed = ConvertSubprocessor ( 
                    converter              = converter.copy_and_set_predictor( model_process_predictor(model_sq,model_cq,model_lock) ), 
                    input_path_image_paths = Path_utils.get_image_paths(input_path), 
                    output_path            = output_path,
                    alignments             = alignments,                                     
                    debug                  = debug ).process()
                              
        model_sq.put ( {'op':'close'} )
        model_p.join()
        
    except Exception as e:
        print ( 'Error: %s' % (str(e)))
        traceback.print_exc()
    
   
