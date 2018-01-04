# Based on the https://github.com/shaoanlu/faceswap-GAN repo (master/FaceSwap_GAN_v2_train.ipynb)

# # 12. Make video clips
# 
# Given a video as input, the following cells will detect face for each frame using dlib's cnn model. And use trained GAN model to transform detected face into target face. Then output a video with swapped faces.

# Download ffmpeg if need, which is required by moviepy.

#import imageio
#imageio.plugins.ffmpeg.download()

# # 13. Make video clips w/o face alignment


import face_recognition
from moviepy.editor import VideoFileClip

use_smoothed_mask = True
use_smoothed_bbox = True

def get_smoothed_coord(x0, x1, y0, y1):
    global prev_x0, prev_x1, prev_y0, prev_y1
    x0 = int(0.65*prev_x0 + 0.35*x0)
    x1 = int(0.65*prev_x1 + 0.35*x1)
    y1 = int(0.65*prev_y1 + 0.35*y1)
    y0 = int(0.65*prev_y0 + 0.35*y0)
    return x0, x1, y0, y1    
    
def set_global_coord(x0, x1, y0, y1):
    global prev_x0, prev_x1, prev_y0, prev_y1
    prev_x0 = x0
    prev_x1 = x1
    prev_y1 = y1
    prev_y0 = y0

def process_video(input_img):   
    # modify this line to reduce input size
    #input_img = input_img[:, input_img.shape[1]//3:2*input_img.shape[1]//3,:] 
    image = input_img
    faces = face_recognition.face_locations(image, model="cnn")
    
    if len(faces) == 0:
        comb_img = np.zeros([input_img.shape[0], input_img.shape[1]*2,input_img.shape[2]])
        comb_img[:, :input_img.shape[1], :] = input_img
        comb_img[:, input_img.shape[1]:, :] = input_img
        triple_img = np.zeros([input_img.shape[0], input_img.shape[1]*3,input_img.shape[2]])
        triple_img[:, :input_img.shape[1], :] = input_img
        triple_img[:, input_img.shape[1]:input_img.shape[1]*2, :] = input_img      
        triple_img[:, input_img.shape[1]*2:, :] = (input_img * .15).astype('uint8')
    
    mask_map = np.zeros_like(image)
    
    global prev_x0, prev_x1, prev_y0, prev_y1
    global frames    
    for (x0, y1, x1, y0) in faces:
        h = x1 - x0
        w = y1 - y0
        
        # smoothing bounding box
        if use_smoothed_bbox:
            if frames != 0:
                x0, x1, y0, y1 = get_smoothed_coord(x0, x1, y0, y1)
                set_global_coord(x0, x1, y0, y1)
            else:
                set_global_coord(x0, x1, y0, y1)
                frames += 1
            
        cv2_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        roi_image = cv2_img[x0+h//15:x1-h//15,y0+w//15:y1-w//15,:]
        roi_size = roi_image.shape  
        
        # smoothing mask
        if use_smoothed_mask:
            mask = np.zeros_like(roi_image)
            mask[h//15:-h//15,w//15:-w//15,:] = 255
            mask = cv2.GaussianBlur(mask,(15,15),10)
            orig_img = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        
        ae_input = cv2.resize(roi_image, (64,64))/255. * 2 - 1        
        result = np.squeeze(np.array([path_abgr_A([[ae_input]])])) # Change path_A/path_B here
        result_a = result[:,:,0] * 255
        result_bgr = np.clip( (result[:,:,1:] + 1) * 255 / 2, 0, 255 )
        result_a = cv2.GaussianBlur(result_a ,(7,7),6)
        result_a = np.expand_dims(result_a, axis=2)
        result = (result_a/255 * result_bgr + (1 - result_a/255) * ((ae_input + 1) * 255 / 2)).astype('uint8')
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        mask_map[x0+h//15:x1-h//15, y0+w//15:y1-w//15,:] = np.expand_dims(cv2.resize(result_a, (roi_size[1],roi_size[0])), axis=2)
        mask_map = np.clip(mask_map + .15 * input_img, 0, 255 )
        
        result = cv2.resize(result, (roi_size[1],roi_size[0]))
        comb_img = np.zeros([input_img.shape[0], input_img.shape[1]*2,input_img.shape[2]])
        comb_img[:, :input_img.shape[1], :] = input_img
        comb_img[:, input_img.shape[1]:, :] = input_img
        
        if use_smoothed_mask:
            comb_img[x0+h//15:x1-h//15, input_img.shape[1]+y0+w//15:input_img.shape[1]+y1-w//15,:] = mask/255*result + (1-mask/255)*orig_img
        else:
            comb_img[x0+h//15:x1-h//15, input_img.shape[1]+y0+w//15:input_img.shape[1]+y1-w//15,:] = result
            
        triple_img = np.zeros([input_img.shape[0], input_img.shape[1]*3,input_img.shape[2]])
        triple_img[:, :input_img.shape[1]*2, :] = comb_img
        triple_img[:, input_img.shape[1]*2:, :] = mask_map
    
    # ========== Change rthe following line to ==========
    # return comb_img[:, input_img.shape[1]:, :]  # return only result image
    # return comb_img  # return input and result image combined as one
    return triple_img #return input,result and mask heatmap image combined as one


# In[ ]:


# Variables for smoothing bounding box
global prev_x0, prev_x1, prev_y0, prev_y1
global frames
prev_x0 = prev_x1 = prev_y0 = prev_y1 = 0
frames = 0

output = 'OUTPUT_VIDEO.mp4'
clip1 = VideoFileClip("INPUT_VIDEO.mp4")
clip = clip1.fl_image(process_video)#.subclip(11, 13) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'clip.write_videofile(output, audio=False)')


# ### gc.collect() sometimes solves memory error

# In[103]:


import gc
gc.collect()

