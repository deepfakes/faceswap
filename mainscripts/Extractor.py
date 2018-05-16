import traceback
import os
import sys
import time
import multiprocessing
from tqdm import tqdm
from pathlib import Path
import numpy as np
import cv2
from utils import Path_utils
from utils.AlignedPNG import AlignedPNG
from utils import image_utils
from facelib import FaceType
import facelib 
import gpufmkmgr
         
from utils.SubprocessorBase import SubprocessorBase
class ExtractSubprocessor(SubprocessorBase):

    #override
    def __init__(self, input_data, type, image_size, face_type, debug, multi_gpu=False, manual=False, detector=None, output_path=None ): 
        self.input_data = input_data
        self.type = type
        self.image_size = image_size
        self.face_type = face_type
        self.debug = debug        
        self.multi_gpu = multi_gpu
        self.detector = detector
        self.output_path = output_path        
        self.manual = manual        
        self.result = []       

        no_response_time_sec = 60 if not self.manual else 999999
        super().__init__('Extractor', no_response_time_sec)           

    #override
    def onHostClientsInitialized(self):
        if self.manual == True:
            self.wnd_name = 'Manual pass'
            cv2.namedWindow(self.wnd_name)
            
            self.view_scale_to = 0 #1368
            
            self.landmarks = None
            self.param_x = -1
            self.param_y = -1
            self.param_rect_size = -1
            self.param = {'x': 0, 'y': 0, 'rect_size' : 5}
                
            def onMouse(event, x, y, flags, param):        
                if event == cv2.EVENT_MOUSEWHEEL:
                    mod = 1 if flags > 0 else -1            
                    param['rect_size'] = max (5, param['rect_size'] + 10*mod)
                else:
                    param['x'] = x
                    param['y'] = y
                    
            cv2.setMouseCallback(self.wnd_name, onMouse, self.param)
    
    def get_devices_for_type (self, type, multi_gpu):
        if (type == 'rects' or type == 'landmarks'):
            if not multi_gpu:            
                devices = [gpufmkmgr.getBestDeviceIdx()]
            else:
                devices = gpufmkmgr.getDevicesWithAtLeastTotalMemoryGB(2)
            devices = [ (idx, gpufmkmgr.getDeviceName(idx), gpufmkmgr.getDeviceVRAMTotalGb(idx) ) for idx in devices]

        elif type == 'final':
            devices = [ (i, 'CPU%d' % (i), 0 ) for i in range(0, multiprocessing.cpu_count()) ]
            
        return devices 
        
    #override
    def process_info_generator(self):    
        for (device_idx, device_name, device_total_vram_gb) in self.get_devices_for_type(self.type, self.multi_gpu): 
            num_processes = 1
            if not self.manual and self.type == 'rects' and self.detector == 'mt':
                num_processes = int ( max (1, device_total_vram_gb / 2) )
                
            for i in range(0, num_processes ):
                device_name_for_process = device_name if num_processes == 1 else '%s #%d' % (device_name,i)
                yield device_name_for_process, {}, {'type' : self.type, 
                                                    'device_idx' : device_idx,
                                                    'device_name' : device_name_for_process, 
                                                    'image_size': self.image_size, 
                                                    'face_type': self.face_type, 
                                                    'debug': self.debug, 
                                                    'output_dir': str(self.output_path), 
                                                    'detector': self.detector}

    #override
    def get_no_process_started_message(self):
        if (self.type == 'rects' or self.type == 'landmarks'):
            print ( 'You have no capable GPUs. Try to close programs which can consume VRAM, and run again.')
        elif self.type == 'final':
            print ( 'Unable to start CPU processes.')
        
    #override
    def onHostGetProgressBarDesc(self):
        return None
        
    #override
    def onHostGetProgressBarLen(self):
        return len (self.input_data)
        
    #override
    def onHostGetData(self):
        if not self.manual:
            if len (self.input_data) > 0:
                return self.input_data.pop(0)    
        else:
            while len (self.input_data) > 0:
                data = self.input_data[0]
                filename, faces = data
                is_frame_done = False
                if len(faces) == 0:
                    self.original_image = cv2.imread(filename)
                    
                    (h,w,c) = self.original_image.shape
                    self.view_scale = 1.0 if self.view_scale_to == 0 else self.view_scale_to / (w if w > h else h)
                    self.original_image = cv2.resize (self.original_image, ( int(w*self.view_scale), int(h*self.view_scale) ), interpolation=cv2.INTER_LINEAR)    
                    
                    self.text_lines_img = (image_utils.get_draw_text_lines ( self.original_image, (0,0, self.original_image.shape[1], min(100, self.original_image.shape[0]) ),
                                                    [   'Match landmarks with face exactly.',
                                                        '[Enter] - confirm frame',
                                                        '[Space] - skip frame',
                                                        '[Mouse wheel] - change rect'
                                                    ], (1, 1, 1) )*255).astype(np.uint8)           

                    while True:
                        key = cv2.waitKey(1) & 0xFF
                        
                        if key == ord('\r') or key == ord('\n'):
                            faces.append ( [(self.rect), self.landmarks] )
                            is_frame_done = True
                            break
                        elif key == ord(' '):
                            is_frame_done = True
                            break
                            
                        if self.param_x != self.param['x'] / self.view_scale or \
                           self.param_y != self.param['y'] / self.view_scale or \
                           self.param_rect_size != self.param['rect_size']:
                           
                            self.param_x = self.param['x'] / self.view_scale
                            self.param_y = self.param['y'] / self.view_scale
                            self.param_rect_size = self.param['rect_size']

                            self.rect = (self.param_x-self.param_rect_size, self.param_y-self.param_rect_size, self.param_x+self.param_rect_size, self.param_y+self.param_rect_size)
                            return [filename, [self.rect]]
                            
                else:
                    is_frame_done = True
                    
                if is_frame_done:
                    self.result.append ( data )
                    self.input_data.pop(0)
                    self.inc_progress_bar(1)

        return None
    
    #override
    def onHostDataReturn (self, data):
        if not self.manual:
            self.input_data.insert(0, data)   
        
    #override
    def onClientInitialize(self, client_dict):
        self.safe_print ('Running on %s.' % (client_dict['device_name']) )
        self.type         = client_dict['type']
        self.image_size   = client_dict['image_size']
        self.face_type    = client_dict['face_type']
        self.device_idx   = client_dict['device_idx']
        self.output_path  = Path(client_dict['output_dir']) if 'output_dir' in client_dict.keys() else None        
        self.debug        = client_dict['debug']
        self.detector     = client_dict['detector']

        self.keras = None
        self.tf = None
        self.tf_session = None
        
        self.e = None
        if self.type == 'rects':
            if self.detector is not None:
                if self.detector == 'mt':
                    self.tf = gpufmkmgr.import_tf ([self.device_idx], allow_growth=True)
                    self.tf_session = gpufmkmgr.get_tf_session()
                    self.keras = gpufmkmgr.import_keras()
                    self.e = facelib.MTCExtractor(self.keras, self.tf, self.tf_session)                            
                elif self.detector == 'dlib':
                    self.dlib = gpufmkmgr.import_dlib( self.device_idx )
                    self.e = facelib.DLIBExtractor(self.dlib)
                self.e.__enter__()

        elif self.type == 'landmarks':
            self.tf = gpufmkmgr.import_tf([self.device_idx], allow_growth=True)
            self.tf_session = gpufmkmgr.get_tf_session()
            self.keras = gpufmkmgr.import_keras()
            self.e = facelib.LandmarksExtractor(self.keras)
            self.e.__enter__()
            
        elif self.type == 'final':
            pass
        
        return None

    #override
    def onClientFinalize(self):
        if self.e is not None:
            self.e.__exit__()
        
    #override
    def onClientProcessData(self, data):
        filename_path = Path( data[0] )

        image = cv2.imread( str(filename_path) )
        if image is None:
            print ( 'Failed to extract %s, reason: cv2.imread() fail.' % ( str(filename_path) ) )
        else:
            if self.type == 'rects':
                rects = self.e.extract_from_bgr (image)  
                return [str(filename_path), rects]

            elif self.type == 'landmarks':
                rects = data[1]   
                landmarks = self.e.extract_from_bgr (image, rects)                    
                return [str(filename_path), landmarks]

            elif self.type == 'final':     
                result = []
                faces = data[1]
                
                if self.debug:
                    debug_output_file = '{}_{}'.format( str(Path(str(self.output_path) + '_debug') / filename_path.stem),  'debug.png')
                    debug_image = image.copy()
                    
                for (face_idx, face) in enumerate(faces):                            
                    rect = face[0]
                    image_landmarks = np.array(face[1])
                    image_to_face_mat = facelib.LandmarksProcessor.get_transform_mat (image_landmarks, self.image_size, self.face_type)
                    output_file = '{}_{}{}'.format(str(self.output_path / filename_path.stem), str(face_idx), '.png')

                    if self.debug:
                        facelib.LandmarksProcessor.draw_rect_landmarks (debug_image, rect, image_landmarks, self.image_size, self.face_type)

                    face_image = cv2.warpAffine(image, image_to_face_mat, (self.image_size, self.image_size))
                    face_image_landmarks = facelib.LandmarksProcessor.transform_points (image_landmarks, image_to_face_mat)
                    
                    cv2.imwrite(output_file, face_image)
                    
                    a_png = AlignedPNG.load (output_file)
                    
                    d = {
                      'face_type': FaceType.toString(self.face_type),
                      'landmarks': face_image_landmarks.tolist(),
                      'yaw_value': facelib.LandmarksProcessor.calc_face_yaw (face_image_landmarks),
                      'pitch_value': facelib.LandmarksProcessor.calc_face_pitch (face_image_landmarks),
                      'source_filename': filename_path.name,
                      'source_rect': rect,
                      'source_landmarks': image_landmarks.tolist()
                    }
                    a_png.setFaceswapDictData (d)
                    a_png.save(output_file)  
                        
                    result.append (output_file)
                    
                if self.debug:
                    cv2.imwrite(debug_output_file, debug_image )
                    
                return result       
        return None
        
    #override
    def onHostResult (self, data, result):
        if self.manual == True:
            self.landmarks = result[1][0][1]
                                        
            image = cv2.addWeighted (self.original_image,1.0,self.text_lines_img,1.0,0)                    
            view_rect = (np.array(self.rect) * self.view_scale).astype(np.int).tolist()
            view_landmarks  = (np.array(self.landmarks) * self.view_scale).astype(np.int).tolist()
            facelib.LandmarksProcessor.draw_rect_landmarks (image, view_rect, view_landmarks, self.image_size, self.face_type)
            
            cv2.imshow (self.wnd_name, image)
            return 0
        else:
            if self.type == 'rects':
                self.result.append ( result )
            elif self.type == 'landmarks':
                self.result.append ( result )                        
            elif self.type == 'final':
                self.result += result
                         
            return 1
    
    #override    
    def onHostProcessEnd(self):
        if self.manual == True:
            cv2.destroyAllWindows()
             
    #override
    def get_start_return(self):
        return self.result

'''
detector
    'dlib'
    'mt'
    'manual'

face_type
    'full_face'
    'avatar'
'''
def main (input_dir, output_dir, debug, detector='mt', multi_gpu=True, manual_fix=False, image_size=256, face_type='full_face'):
    print ("Running extractor.\r\n")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    face_type = FaceType.fromString(face_type)
    
    if not input_path.exists():
        print('Input directory not found. Please ensure it exists.')
        return
        
    if output_path.exists():
        for filename in Path_utils.get_image_paths(output_path):
            Path(filename).unlink()
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        
    if debug:
        debug_output_path = Path(str(output_path) + '_debug')
        if debug_output_path.exists():
            for filename in Path_utils.get_image_paths(debug_output_path):
                Path(filename).unlink()
        else:
            debug_output_path.mkdir(parents=True, exist_ok=True)

    input_path_image_paths = Path_utils.get_image_paths(input_path)
    images_found = len(input_path_image_paths)
    faces_detected = 0
    if images_found != 0:    
        if detector == 'manual':
            print ('Performing manual extract...')
            extracted_faces = ExtractSubprocessor ([ (filename,[]) for filename in input_path_image_paths ], 'landmarks', image_size, face_type, debug, manual=True).process()
        else:
            print ('Performing 1st pass...')
            extracted_rects = ExtractSubprocessor ([ (x,) for x in input_path_image_paths ], 'rects', image_size, face_type, debug, multi_gpu=multi_gpu, manual=False, detector=detector).process()
                
            print ('Performing 2nd pass...')
            extracted_faces = ExtractSubprocessor (extracted_rects, 'landmarks', image_size, face_type, debug, multi_gpu=multi_gpu, manual=False).process()
                
            if manual_fix:
                print ('Performing manual fix...')
                
                if all ( np.array ( [ len(data[1]) > 0 for data in extracted_faces] ) == True ):
                    print ('All faces are detected, manual fix not needed.')
                else:
                    extracted_faces = ExtractSubprocessor (extracted_faces, 'landmarks', image_size, face_type, debug, manual=True).process()

        if len(extracted_faces) > 0:
            print ('Performing 3rd pass...')
            final_imgs_paths = ExtractSubprocessor (extracted_faces, 'final', image_size, face_type, debug, multi_gpu=multi_gpu, manual=False, output_path=output_path).process()
            faces_detected = len(final_imgs_paths)
            
    print('-------------------------')
    print('Images found:        %d' % (images_found) )
    print('Faces detected:      %d' % (faces_detected) )
    print('-------------------------')