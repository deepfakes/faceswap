import sys
import traceback
import queue
import colorsys  
import time
import numpy as np
import itertools

from pathlib import Path
from utils import Path_utils
from utils import image_utils    
import cv2

def trainerThread (input_queue, output_queue, training_data_src_dir, training_data_dst_dir, model_path, model_name, save_interval_min=10, debug=False, target_epoch=0, **in_options):

    while True:
        try: 
            training_data_src_path = Path(training_data_src_dir)
            training_data_dst_path = Path(training_data_dst_dir)
            model_path = Path(model_path)
            
            if not training_data_src_path.exists():
                print( 'Training data src directory is not exists.')
                return
                
            if not training_data_dst_path.exists():
                print( 'Training data dst directory is not exists.')
                return
                
            if not model_path.exists():
                model_path.mkdir(exist_ok=True)
                
            

            import models 
            model = models.import_model(model_name)(
                        model_path, 
                        training_data_src_path=training_data_src_path, 
                        training_data_dst_path=training_data_dst_path, 
                        debug=debug,
                        **in_options)
            
            is_reached_goal = (target_epoch > 0 and model.get_epoch() >= target_epoch)
            
            def model_save():
                if not debug and not is_reached_goal:
                    model.save()
            
            def send_preview():
                if not debug:                        
                    previews = model.get_previews()                
                    output_queue.put ( {'op':'show', 'previews': previews, 'epoch':model.get_epoch(), 'loss_history': model.get_loss_history().copy() } )
                else:
                    previews = [( 'debug, press update for new', model.debug_one_epoch())]
                    output_queue.put ( {'op':'show', 'previews': previews} )
            
            
            if model.is_first_run():
                model_save()
                
            if target_epoch != 0:
                if is_reached_goal:
                    print ('Model already trained to target epoch. You can use preview.')
                else:
                    print('Starting. Target epoch: %d. Press "Enter" to stop training and save model.' % (target_epoch) )
            else: 
                print('Starting. Press "Enter" to stop training and save model.')
 
            last_save_time = time.time()
            for i in itertools.count(0,1):
                if not debug:
                    if not is_reached_goal:
                        loss_string = model.train_one_epoch()     

                        print (loss_string, end='\r')
                        if target_epoch != 0 and model.get_epoch() >= target_epoch:
                            print ('Reached target epoch.')
                            model_save()
                            is_reached_goal = True
                            print ('You can use preview now.')

                if not is_reached_goal and (time.time() - last_save_time) >= save_interval_min*60:
                    last_save_time = time.time() 
                    model_save()
                    send_preview()
                    
                if i==0:
                    if is_reached_goal:
                        model.pass_one_epoch()    
                    send_preview()
                    
                if debug:
                    time.sleep(0.005)
                    
                while not input_queue.empty():
                    input = input_queue.get()
                    op = input['op']
                    if op == 'save':
                        model_save()
                    elif op == 'preview':                    
                        if is_reached_goal:
                            model.pass_one_epoch()                    
                        send_preview()
                    elif op == 'close':
                        model_save()
                        i = -1
                        break
                        
                if i == -1:
                    break
                    
                

            model.finalize()
                
        except Exception as e:
            print ('Error: %s' % (str(e)))
            traceback.print_exc()
        break
    output_queue.put ( {'op':'close'} )

def previewThread (input_queue, output_queue):
    
    
    previews = None
    loss_history = None
    selected_preview = 0
    update_preview = False
    is_showing = False
    is_waiting_preview = False
    epoch = 0
    while True:      
        if not input_queue.empty():
            input = input_queue.get()
            op = input['op']
            if op == 'show':
                is_waiting_preview = False
                loss_history = input['loss_history'] if 'loss_history' in input.keys() else None
                previews = input['previews'] if 'previews' in input.keys() else None
                epoch = input['epoch'] if 'epoch' in input.keys() else 0
                if previews is not None:
                    max_w = 0
                    max_h = 0
                    for (preview_name, preview_rgb) in previews:
                        (h, w, c) = preview_rgb.shape
                        max_h = max (max_h, h)
                        max_w = max (max_w, w)
                    
                    max_size = 800
                    if max_h > max_size:
                        max_w = int( max_w / (max_h / max_size) )
                        max_h = max_size

                    #make all previews size equal
                    for preview in previews[:]:
                        (preview_name, preview_rgb) = preview
                        (h, w, c) = preview_rgb.shape
                        if h != max_h or w != max_w:
                            previews.remove(preview)
                            previews.append ( (preview_name, cv2.resize(preview_rgb, (max_w, max_h))) )
                    selected_preview = selected_preview % len(previews)
                    update_preview = True
            elif op == 'close':
                break
                
        if update_preview:
            update_preview = False
            (h,w,c) = previews[0][1].shape
            
            selected_preview_name = previews[selected_preview][0]
            selected_preview_rgb = previews[selected_preview][1]
            
            # HEAD
            head_text_color = [0.8]*c
            head_lines = [
                '[s]:save [enter]:exit',
                '[p]:update [space]:next preview',
                'Preview: "%s" [%d/%d]' % (selected_preview_name,selected_preview+1, len(previews) )
                ] 
            head_line_height = 15
            head_height = len(head_lines) * head_line_height
            head = np.ones ( (head_height,w,c) ) * 0.1
              
            for i in range(0, len(head_lines)):
                t = i*head_line_height
                b = (i+1)*head_line_height
                head[t:b, 0:w] += image_utils.get_text_image (  (w,head_line_height,c) , head_lines[i], color=head_text_color )
                
            final = head
   
            if loss_history is not None:
                # LOSS HISTORY
                lh_height = 100
                lh_img = np.ones ( (lh_height,w,c) ) * 0.1
                loss_count = len(loss_history[0])
                lh_len = len(loss_history)
                
                l_per_col = lh_len / w                
                plist_max = [   [   max (0.0, 0.0,  *[  loss_history[i_ab][p] 
                                                        for i_ab in range( int(col*l_per_col), int((col+1)*l_per_col) )                                         
                                                     ]
                                        ) 
                                    for p in range(0,loss_count) 
                                ]  
                                for col in range(0, w) 
                            ] 
                            
                            
                plist_min = [   [   min (plist_max[col][p], 
                                         plist_max[col][p],  
                                                    *[  loss_history[i_ab][p] 
                                                        for i_ab in range( int(col*l_per_col), int((col+1)*l_per_col) )                                         
                                                     ]
                                        ) 
                                    for p in range(0,loss_count) 
                                ]  
                                for col in range(0, w) 
                            ] 
                plist_abs_max = 0.03
                
                if l_per_col >= 1.0:
                    for col in range(0, w):
                        for p in range(0,loss_count): 
                            point_color = [1.0]*c
                            point_color[0:3] = colorsys.hsv_to_rgb ( p * (1.0/loss_count), 1.0, 1.0 )
                            
                            ph_max = int ( (plist_max[col][p] / plist_abs_max) * (lh_height-1) )
                            ph_max = np.clip( ph_max, 0, lh_height-1 )
                            
                            ph_min = int ( (plist_min[col][p] / plist_abs_max) * (lh_height-1) )
                            ph_min = np.clip( ph_min, 0, lh_height-1 )
                            
                            for ph in range(ph_min, ph_max+1):
                                lh_img[ (lh_height-ph-1), col ] = point_color
                                
                lh_lines = 5
                lh_line_height = (lh_height-1)/lh_lines
                for i in range(0,lh_lines+1):
                    lh_img[ int(i*lh_line_height), : ] = (0.8,)*c
                    
                last_line_t = int((lh_lines-1)*lh_line_height)
                last_line_b = int(lh_lines*lh_line_height)
                
                if epoch != 0:
                    lh_text = 'Loss history. Epoch: %d' % (epoch)
                else:
                    lh_text = 'Loss history.'
                
                lh_img[last_line_t:last_line_b, 0:w] += image_utils.get_text_image (  (w,last_line_b-last_line_t,c), lh_text, color=head_text_color )
                
                final = np.concatenate ( [final, lh_img], axis=0 )
                
            final = np.concatenate ( [final, selected_preview_rgb], axis=0 )
            
            cv2.imshow ( 'Training preview', final)
            is_showing = True
        
        if is_showing:
            key = cv2.waitKey(100)
        else:
            time.sleep(0.1)
            key = 0

        if key == ord('\n') or key == ord('\r'):
            output_queue.put ( {'op': 'close'} )
        elif key == ord('s'):
            output_queue.put ( {'op': 'save'} )
        elif key == ord('p'):
            if not is_waiting_preview:
                is_waiting_preview = True
                output_queue.put ( {'op': 'preview'} )
        elif key == ord(' '):
            selected_preview = (selected_preview + 1) % len(previews)
            update_preview = True
            
    cv2.destroyAllWindows()
    
def main (training_data_src_dir, training_data_dst_dir, model_path, model_name, **in_options):
    print ("Running trainer.\r\n")
    
    output_queue = queue.Queue()
    input_queue = queue.Queue()    
    import threading
    thread = threading.Thread(target=trainerThread, args=(output_queue, input_queue, training_data_src_dir, training_data_dst_dir, model_path, model_name), kwargs=in_options )
    thread.start()
    
    previewThread (input_queue, output_queue)