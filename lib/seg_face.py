import keras
from keras.models import Model, load_model
from keras import layers
import keras.backend as K
import numpy
import cv2
import os,sys,inspect,gc,pathlib
import pydensecrf
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from multithreading import BackgroundGenerator
from utils import parse_model_weights

class Mask():
    def __init__(self):
        image_size = (256,256)
        batch_size = 24
        image_directory = pathlib.Path('C:\data\images')
        model_file = 'C:/data/face_seg_300/converted_model.h5'
        
        #weight_file = pathlib.Path(r'C:\data\face_seg_300\converted_caffe_IR.npy')
        #model = self.parse_model_weights(weight_file, model_file=model_file)
        
        model = load_model(model_file)
        print('\n' + 'Model loaded')
        
        image_file_list = self.get_image_paths(image_directory) 
        num_of_batches = len(image_file_list) // batch_size + 1
        image_dataset = self.dataset_setup(image_file_list, image_size, batch_size)
        image_generator = self.minibatches(image_dataset)
        print('\n' + 'Image dataset & generator loaded')
        
        print('\n' + 'Predicting batches of images in model')
        i=0
        for num, batches in enumerate(range(num_of_batches)):
            batch_of_images = next(image_generator)    
            batch_of_results = model.predict_on_batch(batch_of_images)
            print('   --- Batch number ' + str(num) + ': ---')
            print('       - model run complete')
            
            batch_of_masks = self.postprocessing(batch_of_results, batch_of_images, image_size)
            blended = self.blend_image_and_mask(batch_of_images,batch_of_masks)
            print('       - postprocessing complete')
            
            for mask in blended:
            #for mask in resized_masks:
                if i < len(image_file_list):
                    p = pathlib.Path(image_file_list[i])
                    cv2.imwrite(str(image_directory) + r'\mask\ ' + str(p.stem) + '.png', mask)
                    i += 1
            print('       - masks saved to directory')
            
        print('\n' + 'Mask generation complete')
           
    def get_image_paths(self, directory):
        image_extensions = [".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"]
        dir_contents = list()
        dir_scanned = sorted(os.scandir(directory), key=lambda x: x.name)

        for chkfile in dir_scanned:
            if any([chkfile.name.lower().endswith(ext) for ext in image_extensions]):
                dir_contents.append(chkfile.path)
                
        return dir_contents
        
    def dataset_setup(self, image_list, image_size, batch_size):
        # create a .npy file in  image folder 
        # the .npy memmapped dataset has dimensions (batch_num,batch_size,256,256,3)
        # iterate/load images them into the .npy dataset using opencv
        length = len(image_list)
        extra = 0 if (length % batch_size) == 0 else 1
        batch_num = length // batch_size + extra
        filename = str(pathlib.Path(image_list[0]).parents[0].joinpath(('Images_Batched(' + str(batch_size) + ').npy')))
        dataset=numpy.lib.format.open_memmap(filename, mode='w+',
                                             dtype='uint8',
                                             shape=(batch_num,batch_size,
                                                    image_size[0],image_size[1],3))
        for i, image_array in enumerate((cv2.imread(image_file) for image_file in image_list)):
            dataset[(i-1)//batch_size, (i-1)%batch_size] = image_array
        del dataset
        gc.collect()
        
        return filename
        
    def minibatches(self, filename):
        # create a generator that iterates over a memmapped image dataset
        memmapped_images = numpy.load(filename, mmap_mode='r+')
        yield from BackgroundGenerator(self.generate(memmapped_images),1).iterator()
        
    def generate(self, memmapped_images):
        while True:
            # Iterate through the dataset entire batches at a time 
            # Call the process_images function on each batch
            yield from (self.process_images(batch.astype('float32')) for batch in memmapped_images[:])
            
    def process_images(self, batch):
        images = numpy.empty((batch.shape[0], 300, 300, batch.shape[3]),
                              dtype='float32')
        for i, image in enumerate(batch):
            images[i] = self.preprocessing(image)
        return images

    def blend_image_and_mask(self, image_batch, mask_batch, alpha=0.5, color=(0.0,0.0,127.5)):
        image_batch += numpy.array((104.00698793,116.66876762,122.67891434))
        mask_batch = numpy.repeat(mask_batch, 3, axis=-1)
        mask_batch *= image_batch
        image_batch = numpy.concatenate((image_batch, mask_batch),axis = 2)
        return image_batch.astype('uint8')

    def preprocessing(self, image):
        # https://github.com/YuvalNirkin/find_face_landmarks/blob/master/interfaces/matlab/bbox_from_landmarks.m
        # input images should be cropped like this for best results
        
        image_size = image.shape[0],image.shape[1]
        if image.shape[0] != 300:
            interpolator = cv2.INTER_CUBIC if image.shape[0] / 300 > 1.0 else cv2.INTER_AREA
            image = cv2.resize(image, (300,300), interpolator)
        image -= numpy.array((104.00698793,116.66876762,122.67891434))
        #image -= numpy.average(image, axis=(0,1))
        image = numpy.expand_dims(image, 0)
        
        return image
    
    def crop_standarization(self,image,landmarks):
        (box_x, box_y), (w, h), rect_angle = cv2.boundingRect(landmarks)
        w, h = max(w, h), max(w, h)
        M = cv2.moments(landmarks)
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        
        w = w * 1.2 + abs(box_x - center_x)
        h = h * 1.2 + abs(box_y - center_x)
        
        xmin = int(max(0, box_x - w / 2.0))
        ymin = int(max(0, box_y - h / 2.0))
        xmax = int(min(image.shape[1], box_x + w / 2.0))
        ymax = int(min(image.shape[0], box_y + h / 2.0))
        rect = slice(ymin,ymax),slice(xmin,xmax)
        
        return rect
        
    def postprocessing(self, batch_of_results,batch_of_images, image_size):
        batches = zip(batch_of_results,batch_of_images)
        mask_list = [self.dense_crf(mask,image) for mask, image in batches]
        mask_list = [cv2.resize(mask, image_size, cv2.INTER_CUBIC) for mask in mask_list]
        resized_batch = numpy.stack(mask_list,axis=0)
        resized_batch = resized_batch.argmax(axis=3).astype('float32')
        resized_batch = numpy.clip(resized_batch,0.0,1.0)
        resized_batch = numpy.expand_dims(resized_batch, axis=-1)
        #resized_batch[resized_batch!=0.0] = 1.0
        resized_batch = self.fill_holes(resized_batch)
        #resized_batch = self.smooth_contours(resized_batch)
        #resized_batch = self.select_largest_segment(resized_batch)
        
        return resized_batch

    def select_largest_segment(self, mask):
        results = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
        num_labels, labels, stats, centroids = results
        segments_ranked_by_area = numpy.argsort(stats[:,-1])[::-1]
        mask[labels != segments_ranked_by_area[0,0]] = 0.0
        
        return mask
        
    def smooth_contours(self, mask, smooth_iterations=2, smooth_kernel_radius=2):
        kernel_size = int(smooth_kernel_radius * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (kernel_size, kernel_size))
        for i in range(smooth_iterations):
            cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, anchor=(-1, -1),
                             iterations=smooth_iterations)
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, anchor=(-1, -1),
                             iterations=smooth_iterations)
        
        return mask
        
    def fill_holes(self, mask):
        black_background = numpy.zeros((mask.shape[0]+4,mask.shape[1]+4,
                                       mask.shape[2]), dtype = 'float32')
        central = slice(2,-2), slice(2,-2)
        black_background[central] = mask
        cv2.floodFill(black_background, None, (0, 0), 1.0)
        mask[black_background[central]==0.0] = 1.0
        
        return mask

    def dense_crf(self, probs, img, n_iters=10, 
                  sxy_gaussian=(1, 1), compat_gaussian=4,
                  kernel_gaussian=pydensecrf.densecrf.DIAG_KERNEL,
                  normalisation_gaussian=pydensecrf.densecrf.NORMALIZE_SYMMETRIC,
                  sxy_bilateral=(50, 50), compat_bilateral=5,
                  srgb_bilateral=(12, 12, 12),
                  kernel_bilateral=pydensecrf.densecrf.DIAG_KERNEL,
                  normalisation_bilateral=pydensecrf.densecrf.NORMALIZE_SYMMETRIC):
        """DenseCRF over unnormalised predictions.
           More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.
        
        Args:
          probs: class probabilities per pixel.
          img: if given, the pairwise bilateral potential on raw RGB values will be computed.
          n_iters: number of iterations of MAP inference.
          sxy_gaussian: standard deviations for the location component of the colour-independent term.
          compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
          kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
          normalisation_gaussian: normalisation for the colour-independent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
          sxy_bilateral: standard deviations for the location component of the colour-dependent term.
          compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
          srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
          kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
          normalisation_bilateral: normalisation for the colour-dependent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
          
        Returns:
          Refined predictions after MAP inference.
        """
        h, w, _ = probs.shape
        
        probs = probs.transpose(2, 0, 1).copy(order='C') # Need a contiguous array.
        
        d = pydensecrf.densecrf.DenseCRF2D(w, h, n_classes) # Define DenseCRF model.
        U = -np.log(probs) # Unary potential.
        U = U.reshape((n_classes, -1)) # Needs to be flat.
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                              kernel=kernel_gaussian, normalization=normalisation_gaussian)
        assert(img.shape[0:2] == (h, w)), "The image height and width must coincide with dimensions of the logits."
        d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
                               kernel=kernel_bilateral, normalization=normalisation_bilateral,
                               srgb=srgb_bilateral, rgbim=img)
        Q = d.inference(n_iters)
        preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w)).transpose(1, 2, 0)
        return preds

if __name__ == '__main__':
    Mask()