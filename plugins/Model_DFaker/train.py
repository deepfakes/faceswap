import cv2
import numpy

from utils import get_image_paths, load_images, stack_images
from training_data import get_training_data
import glob

import random

from scipy.stats import linregress
from tqdm import tqdm

if __name__ == '__main__':

#   print('running')

#   images_A = get_image_paths( "data/A" )
#   images_B = get_image_paths( "data/B"  )

#   minImages = 2000#min(len(images_A),len(images_B))*20

#   random.shuffle(images_A)
#   random.shuffle(images_B)

#   images_A,landmarks_A = load_images( images_A[:minImages] ) 
#   images_B,landmarks_B = load_images( images_B[:minImages] )

#   print('Images A', images_A.shape)
#   print('Images B', images_B.shape)

#   images_A = images_A/255.0
#   images_B = images_B/255.0

#   images_A[:,:,:3] += images_B[:,:,:3].mean( axis=(0,1,2) ) - images_A[:,:,:3].mean( axis=(0,1,2) )

#   print( "press 'q' to stop training and save model" )

#   batch_size = int(32)

#   warped_A, target_A, mask_A = get_training_data( images_A,  landmarks_A,landmarks_B, batch_size )
#   warped_B, target_B, mask_B  = get_training_data( images_B, landmarks_B,landmarks_A, batch_size )


#   print(warped_A.shape, target_A.shape, mask_A.shape)

#   figWarped = numpy.stack([warped_A[:6],warped_B[:6]],axis=0 )
#   figWarped = numpy.clip( figWarped * 255, 0, 255 ).astype('uint8')
#   figWarped = stack_images( figWarped )
#   cv2.imshow( "w", figWarped )

#   print(warped_A.shape)
#   print(target_A.shape)

##### => moved to AutoEncoder

  while 1:
    pbar = tqdm(range(1000000))
    for epoch in pbar:

      

 
        pbar.set_description("Loss A [{}] Loss B [{}]".format(loss_A,loss_B))


        if epoch % 100 == 0:
          save_model_weights()
          test_A = target_A[0:8,:,:,:3]
          test_B = 








        # cv2.imshow( "p", figure )
    
        # key = cv2.waitKey(1)
        # if key == ord('q'):
        #     save_model_weights()
        #     exit()

