import os
import sys
from tqdm import tqdm

from utils import Path_utils
from utils.AlignedPNG import AlignedPNG
from utils import image_utils
from facelib import LandmarksProcessor
from pathlib import Path

import cv2
import traceback
import multiprocessing
import time

import facelib 
import gpufmkmgr
import numpy as np

def extract_pass_process(sq, cq):
    e = None
    type = None
    device_idx = None
    debug = False
    output_path = None
    detector = None
    image_size = None
    face_type = None
    while True:
        obj = sq.get()
        obj_op = obj['op']

        if obj_op == 'extract':
            data = obj['data']

            filename_path = Path( data[0] )

            if not filename_path.exists():
                cq.put ( {'op': 'error', 'close': False, 'message': 'Failed to extract %s, reason: file not found.' % ( str(filename_path) ) } )
            else:                
                try:
                    image = cv2.imread( str(filename_path) )
                    if image is None:
                        cq.put ( {'op': 'error', 'close': False, 'message': 'Failed to extract %s, reason: cv2.imread() fail.' % ( str(filename_path) ) } )
                    else:
                        if type == 'rects':
                            rects = e.extract_from_bgr (image)    
                            cq.put ( {'op': 'extract_success', 'data' : obj['data'], 'result' : [str(filename_path), rects] } )
                            
                        elif type == 'landmarks':
                            rects = data[1]   
                            landmarks = e.extract_from_bgr (image, rects)                    
                            cq.put ( {'op': 'extract_success', 'data' : obj['data'], 'result' : [str(filename_path), landmarks] } )

                        elif type == 'final':     
                            result = []
                            faces = data[1]
                            
                            if debug:
                                debug_output_file = '{}_{}'.format( str(Path(str(output_path) + '_debug') / filename_path.stem),  'debug.png')
                                debug_image = image.copy()
                                
                            for (face_idx, face) in enumerate(faces):                            
                                rect = face[0]
                                image_landmarks = np.array(face[1])
                                image_to_face_mat = facelib.LandmarksProcessor.get_transform_mat (image_landmarks, image_size, face_type)
                                output_file = '{}_{}{}'.format(str(output_path / filename_path.stem), str(face_idx), '.png')

                                if debug:
                                    facelib.LandmarksProcessor.draw_rect_landmarks (debug_image, rect, image_landmarks, image_size, face_type)
       
                                face_image = cv2.warpAffine(image, image_to_face_mat, (image_size, image_size))
                                face_image_landmarks = facelib.LandmarksProcessor.transform_points (image_landmarks, image_to_face_mat)
                                
                                cv2.imwrite(output_file, face_image)
                                
                                a_png = AlignedPNG.load (output_file)
                                
                                d = {
                                  'type': 'face',
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
                                
                            if debug:
                                cv2.imwrite(debug_output_file, debug_image )
                                
                            cq.put ( {'op': 'extract_success', 'data' : obj['data'], 'result' : result} )
                    
                except Exception as e:
                    cq.put ( {'op': 'error', 'close': True, 'data' : obj['data'], 'message' : 'Failed to extract %s, reason: %s. \r\n%s' % ( str(filename_path), str(e), traceback.format_exc() ) } )
                    break
                    
        elif obj_op == 'init':
            try:
                type = obj['type']
                image_size = obj['image_size']
                face_type = obj['face_type']
                device_idx = obj['device_idx']
                output_path = Path(obj['output_dir']) if 'output_dir' in obj.keys() else None
                debug = obj['debug']
                detector = obj['detector']
                
                if type == 'rects':
                    if detector is not None:
                        if detector == 'mt':
                            tf = gpufmkmgr.import_tf ([device_idx], allow_growth=True)
                            tf_session = gpufmkmgr.get_tf_session()
                            keras = gpufmkmgr.import_keras()
                            e = facelib.MTCExtractor(keras, tf, tf_session)                            
                        elif detector == 'dlib':
                            dlib = gpufmkmgr.import_dlib( device_idx )
                            e = facelib.DLIBExtractor(dlib)
                        e.__enter__()
                    
                        
                elif type == 'landmarks':
                    gpufmkmgr.import_tf([device_idx], allow_growth=True)
                    keras = gpufmkmgr.import_keras()
                    e = facelib.LandmarksExtractor(keras)
                    e.__enter__()
                elif type == 'final':
                    pass
                
                cq.put ( {'op': 'init_ok'} )
            except Exception as e:                
                cq.put ( {'op': 'error', 'close': True, 'message': 'Exception while initialization: %s' % (traceback.format_exc()) } )
                break
    
    if detector is not None and (type == 'rects' or type == 'landmarks'):
        e.__exit__()        
                
def start_processes ( devices, type, image_size, face_type, debug, detector, output_path ):
    extract_processes = []
    for (device_idx, device_name, device_total_vram_gb) in devices:   
        if type == 'rects':
            if detector == 'mt':
                num_processes = int ( max (1, device_total_vram_gb / 2) )
            else:
                num_processes = 1
        else:
            num_processes = 1
            
        for i in range(0, num_processes ):
            sq = multiprocessing.Queue()
            cq = multiprocessing.Queue()
            p = multiprocessing.Process(target=extract_pass_process, args=(sq,cq))
            p.daemon = True
            p.start()

            sq.put ( {'op': 'init', 'type' : type, 'device_idx' : device_idx, 'image_size':image_size, 'face_type':face_type, 'debug':debug, 'output_dir': str(output_path), 'detector':detector } )
            
            device_name_for_process = device_name if num_processes == 1 else '%s #%d' % (device_name,i)

            extract_processes.append ( {
                'process' : p,
                'device_idx' : device_idx,
                'device_name' : device_name_for_process,
                'sq' : sq,
                'cq' : cq,
                'state' : 'busy',
                'sent_time': time.time(),
                'attempt' : 0
                } )
            
    while True:
        for p in extract_processes[:]:
            while not p['cq'].empty():
                obj = p['cq'].get()
                obj_op = obj['op']
                    
                if obj_op == 'init_ok':
                    print ( 'Running extract on %s.' % ( p['device_name'] ) )      
                elif obj_op == 'error':
                    print (obj['message'])
                    if obj['close'] == True:
                        p['process'].terminate()
                        p['process'].join()
                        extract_processes.remove(p)
                        break                
                p['state'] = 'free'        
        if all ([ p['state'] == 'free' for p in extract_processes ] ):
            break
        
    return extract_processes

def get_devices_for_type (type, multi_gpu):
    if (type == 'rects' or type == 'landmarks'):
        if not multi_gpu:            
            devices = [gpufmkmgr.getBestDeviceIdx()]
        else:
            devices = gpufmkmgr.getDevicesWithAtLeastTotalMemoryGB(2)
        devices = [ (idx, gpufmkmgr.getDeviceName(idx), gpufmkmgr.getDeviceVRAMTotalGb(idx) ) for idx in devices]

    elif type == 'final':
        devices = [ (i, 'CPU%d' % (i), 0 ) for i in range(0, multiprocessing.cpu_count()) ]
        
    return devices    
    
#type='rects 'landmarks' 'final'
def extract_pass(input_data, type, image_size, face_type, debug, multi_gpu, detector=None, output_path=None):
    if type=='landmarks':
        if all ( np.array ( [ len(data[1]) == 0 for data in input_data] ) == True ):
            #no rects - no landmarks
            return [ (data[0], []) for data in input_data]
    
    extract_processes = start_processes (get_devices_for_type(type, multi_gpu), type, image_size, face_type, debug, detector, output_path)
    if len(extract_processes) == 0:
        if (type == 'rects' or type == 'landmarks'):
            print ( 'You have no capable GPUs. Try to close programs which can consume VRAM, and run again.')
        elif type == 'final':
            print ( 'Unable to start CPU processes.')
        return []
            
    result = []
    progress_bar = tqdm( total=len(input_data) )    

    while True:
        for p in extract_processes[:]:
            while not p['cq'].empty():
                obj = p['cq'].get()
                obj_op = obj['op']
                
                if obj_op == 'extract_success':                
                    if type == 'rects':
                        result.append ( obj['result'] )
                    elif type == 'landmarks':
                        result.append ( obj['result'] )                        
                    elif type == 'final':
                        result += obj['result']
                        
                    progress_bar.update(1)

                elif obj_op == 'error':
                    print (obj['message'])
                    if 'data' in obj.keys():
                        filename = obj['data'][0]
                        input_data.insert(0, obj['data'])                    
                    if obj['close'] == True:
                        p['process'].terminate()
                        p['process'].join()
                        extract_processes.remove(p)
                        break
                    
                p['state'] = 'free'
    
        if len(input_data) == 0 and all ([p['state'] == 'free' for p in extract_processes]):
            break

        for p in extract_processes[:]:
            if p['state'] == 'free' and len(input_data) > 0:   
                data = input_data.pop(0)
                p['sq'].put ( {'op': 'extract', 'data' : data} )
                p['sent_time'] = time.time()
                p['sent_data'] = data
                p['state'] = 'busy'
            elif p['state'] == 'busy':
                if (time.time() - p['sent_time']) > 60:
                    print ( '%s doesnt response on %s, terminating it.' % (p['device_name'], p['sent_data'][0]) )
                    input_data.insert(0, p['sent_data'])
                    p['process'].terminate()
                    p['process'].join()
                    extract_processes.remove(p)
           
        time.sleep(0.005)
        
    for p in extract_processes[:]:
        p['process'].terminate()
        p['process'].join()
        
    progress_bar.close()
    return result

def manual_pass( extracted_faces, image_size, face_type ):
    extract_processes = start_processes (get_devices_for_type('landmarks',False), 'landmarks', image_size, face_type, False, None, None)
    if len(extract_processes) == 0:
        if (type == 'rects' or type == 'landmarks'):
            print ('You have no capable GPUs. Try to close programs which can consume VRAM, and run again.')
        return []
        
    p = extract_processes[0]
        
    wnd_name = 'Manual pass'
    cv2.namedWindow(wnd_name)
    
    view_scale_to = 0 #1368
    
    param = {'x': 0, 'y': 0, 'rect_size' : 5}
        
    def onMouse(event, x, y, flags, param):        
        if event == cv2.EVENT_MOUSEWHEEL:
            mod = 1 if flags > 0 else -1            
            param['rect_size'] = max (5, param['rect_size'] + 10*mod)
        else:
            param['x'] = x
            param['y'] = y
            
    cv2.setMouseCallback(wnd_name, onMouse, param)

    for data in extracted_faces:  
        filename = data[0]
        faces = data[1]
        if len(faces) == 0:
            original_image = cv2.imread(filename)
            
            (h,w,c) = original_image.shape
            view_scale = 1.0 if view_scale_to == 0 else view_scale_to / (w if w > h else h)
            original_image = cv2.resize (original_image, ( int(w*view_scale), int(h*view_scale) ), interpolation=cv2.INTER_LINEAR)    
            
            text_lines_img = (image_utils.get_draw_text_lines ( original_image, (0,0, original_image.shape[1], min(100, original_image.shape[0]) ),
                                            [   'Match landmarks with face exactly.',
                                                '[Enter] - confirm frame',
                                                '[Space] - skip frame',
                                                '[Mouse wheel] - change rect'
                                            ], (1, 1, 1) )*255).astype(np.uint8)           
                                
            landmarks = None
            param_x = -1
            param_y = -1
            param_rect_size = -1
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('\r') or key == ord('\n'):
                    faces.append ( [(rect), landmarks] )
                    break
                elif key == ord(' '):
                    break
                    
                if param_x != param['x'] / view_scale or \
                   param_y != param['y'] / view_scale or \
                   param_rect_size != param['rect_size']:
                   
                    param_x = param['x'] / view_scale
                    param_y = param['y'] / view_scale
                    param_rect_size = param['rect_size']

                    rect = (param_x-param_rect_size, param_y-param_rect_size, param_x+param_rect_size, param_y+param_rect_size)
                    p['sq'].put ( {'op': 'extract', 'data' : [filename, [rect]]} )
                    
                    while True:
                        if not p['cq'].empty():
                            obj = p['cq'].get()
                            obj_op = obj['op']           
                            if obj_op == 'extract_success':
                                result = obj['result']
                                landmarks = result[1][0][1]
                                
                                image = cv2.addWeighted (original_image,1.0,text_lines_img,1.0,0)                    
                                view_rect = (np.array(rect) * view_scale).astype(np.int).tolist()
                                view_landmarks  = (np.array(landmarks) * view_scale).astype(np.int).tolist()
                                facelib.LandmarksProcessor.draw_rect_landmarks (image, view_rect, view_landmarks, image_size, face_type)
                                
                                cv2.imshow (wnd_name, image)                            
                            elif obj_op == 'error':
                                print (obj['message'])
                            break                 


    cv2.destroyAllWindows()
    
    for p in extract_processes[:]:
        p['process'].terminate()
        p['process'].join()
                        
    return extracted_faces
 
'''
face_type
    'full_face'
    'head' - unused
'''
def main (input_dir, output_dir, debug, detector='mt', multi_gpu=True, manual_fix=False, image_size=256, face_type='full_face'):
    print ("Running extractor.\r\n")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    

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
    if images_found != 0:
    
        if detector == 'manual':
            print ('Performing manual extract...')
            extracted_faces = [ (filename,[]) for filename in input_path_image_paths ]
            extracted_faces = manual_pass(extracted_faces, image_size, face_type)
        else:
            print ('Performing 1st pass...')
            extracted_rects = extract_pass( [ (x,) for x in input_path_image_paths ], 'rects', image_size, face_type, debug, multi_gpu, detector=detector)

            print ('Performing 2nd pass...')
            extracted_faces = extract_pass(extracted_rects, 'landmarks', image_size, face_type, debug, multi_gpu)
                
            if manual_fix:
                print ('Performing manual fix...')
                
                if all ( np.array ( [ len(data[1]) > 0 for data in extracted_faces] ) == True ):
                    print ('All faces are detected, manual fix not needed.')
                else:
                    extracted_faces = manual_pass(extracted_faces, image_size, face_type)
        
        if len(extracted_faces) > 0:
            print ('Performing 3rd pass...')
            final_imgs_paths = extract_pass (extracted_faces, 'final', image_size, face_type, debug, multi_gpu, image_size, output_path=output_path)
            faces_detected = len(final_imgs_paths)
    else:
        faces_detected = 0
        
    print('-------------------------')
    print('Images found:        %d' % (images_found) )
    print('Faces detected:      %d' % (faces_detected) )
    print('-------------------------')