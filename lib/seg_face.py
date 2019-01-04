import keras
from keras.models import Model, load_model
from keras import layers
import keras.backend as K
import numpy
import cv2
import os,sys,inspect,gc,pathlib
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from multithreading import BackgroundGenerator


class Mask():
    def __init__(self):
        
        #weight_file = pathlib.Path('C:/data/edfb3bac7e774b609fbae1d6ffb68e3d.npy')
        #model = mask_model(weight_file)
        
        model = load_model('C:/data/face_seg_model.h5')
        #model.compile()
        image_directory = pathlib.Path('C:/data/images')
        image_file_list = self.get_image_paths(image_directory) 
        batch_size = 16
        num_of_batches = len(image_file_list) // batch_size + 1
        
        image_dataset = self.dataset_setup(image_file_list, batch_size)
        image_generator = self.minibatches(image_dataset)
        
        for batches in range(num_of_batches):
            batch_of_images = next(image_generator)    
            batch_of_results = model.predict_on_batch(batch_of_images)
            
            batch_of_masks = batch_of_results.argmax(axis=3)
            #batch_of_masks = batch_of_masks.transpose((0,2,3,1))
            print(batch_of_masks)
            print(batch_of_masks.shape)
            mask_list = [self.postprocessing(mask) for mask in batch_of_masks]
            resized_masks = [cv2.resize(mask, image_size, cv2.INTER_NEAREST) for mask in mask_list]
            print('here')
            for i, mask in enumerate(resized_masks):
               cv2.imwrite(str(image_directory) + 'mask-' + str(i) + '.png', mask)
           
    def get_image_paths(self, directory):
        image_extensions = [".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"]
        dir_contents = list()
        dir_scanned = sorted(os.scandir(directory), key=lambda x: x.name)

        for chkfile in dir_scanned:
            if any([chkfile.name.lower().endswith(ext) for ext in image_extensions]):
                dir_contents.append(chkfile.path)
                
        return dir_contents
        
    def dataset_setup(self, image_list, batch_size):
        # create a .npy file in  image folder 
        # the .npy memmapped dataset has dimensions (batch_num,batch_size,256,256,3)
        # iterate/load images them into the .npy dataset using opencv
        length = len(image_list)
        extra = 0 if (length % batch_size) == 0 else 1
        batch_num = length // batch_size + extra
        filename = str(pathlib.Path(image_list[0]).parents[0].joinpath(('Images_Batched(' + str(batch_size) + ').npy')))
        dataset=numpy.lib.format.open_memmap(filename, mode='w+',
                                             dtype=numpy.uint8,
                                             shape=(batch_num,batch_size,256,256,3))
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
        images = numpy.empty((batch.shape[0], 500, 500, batch.shape[3]),
                              dtype='float32')
        for i, image in enumerate(batch):
            images[i] = self.preprocessing(image)
        return images

    def blend_image_and_mask(self, image, mask, alpha, color=[0,0,255]):
        image[mask>128] = int(color * alpha + image * ( 1 - alpha ))
        # image[:,:,:3][mask>128] = int(color * alpha + image * ( 1 - alpha )
        return image

    def preprocessing(self, image):
        # https://github.com/YuvalNirkin/find_face_landmarks/blob/master/interfaces/matlab/bbox_from_landmarks.m
        # input images should be cropped like this for best results

        # resize to 500x500 pixels
        # subtract channel mean
        # put in N,C,H,W tensor format
        image_size = image.shape[0],image.shape[1]
        interpolator = cv2.INTER_CUBIC if image.shape[0] / 500 > 1.0 else cv2.INTER_AREA
        image = cv2.resize(image, (500,500), interpolator)
        image -= numpy.average(image, axis=(0,1))
        #image = image.transpose((2,0,1))
        image = numpy.expand_dims(image, 0)
        
        return image

    def postprocessing(self, mask):
        mask = self.select_largest_segment(mask)
        mask = self.fill_holes(mask)
        mask = self.smooth_flaws(mask)
        mask = self.select_largest_segment(mask)
        mask = self.fill_holes(mask)
        
        return mask

    def select_largest_segment(self, mask):
        results = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
        num_labels, labels, stats, centroids = results
        segments_ranked_by_area = np.argsort(stats[:,-1])[::-1]
        mask[labels != segments_ranked_by_area[0,0]] = 0
        
        return mask
        
    def smooth_flaws(self, mask, smooth_iterations=1, smooth_kernel_radius=2):
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
        mask[mask!=0] = 255
        holes = mask.copy()
        cv2.floodFill(holes, None, (0, 0), 255)
        holes = cv2.bitwise_not(holes)
        filled_mask = cv2.bitwise_or(mask, holes)
        
        return filled_mask
        
    def load_weights_from_file(weight_file):
        try:
            weights_dict = numpy.load(weight_file).item()
        except:
            weights_dict = numpy.load(weight_file, encoding='bytes').item()
            
        return weights_dict
        
    def set_layer_weights(model, weights_dict):
        for layer in model.layers:
            if layer.name in weights_dict:
                cur_dict = weights_dict[layer.name]
                current_layer_parameters = list()
                if layer.__class__.__name__ == "BatchNormalization":
                    if 'scale' in cur_dict:
                        current_layer_parameters.append(cur_dict['scale'])
                    if 'bias' in cur_dict:
                        current_layer_parameters.append(cur_dict['bias'])
                    current_layer_parameters.extend([cur_dict['mean'],
                                                    cur_dict['var']])
                elif layer.__class__.__name__ == "Scale":
                    if 'scale' in cur_dict:
                        current_layer_parameters.append(cur_dict['scale'])
                    if 'bias' in cur_dict:
                        current_layer_parameters.append(cur_dict['bias'])
                elif layer.__class__.__name__ == "SeparableConv2D":
                    current_layer_parameters = [cur_dict['depthwise_filter'],
                                                cur_dict['pointwise_filter']]
                    if 'bias' in cur_dict:
                        current_layer_parameters.append(cur_dict['bias'])
                else:
                    current_layer_parameters = [cur_dict['weights']]
                    if 'bias' in cur_dict:
                        current_layer_parameters.append(cur_dict['bias'])
                model.get_layer(layer.name).set_weights(current_layer_parameters)
                
        return model
        
    def mask_model(weight_file = None):
        weights_dict = load_weights_from_file(weight_file) if not weight_file == None else None
            
        input           = layers.Input(name = 'input', shape = (500, 500, 3,) )
        conv1_1_input   = layers.ZeroPadding2D(padding = ((100, 100), (100, 100)))(input)
        conv1_1         = convolution(weights_dict, name='conv1_1', input=conv1_1_input, group=1, conv_type='layers.Conv2D', filters=64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
        relu1_1         = layers.Activation(name='relu1_1', activation='relu')(conv1_1)
        conv1_2_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu1_1)
        conv1_2         = convolution(weights_dict, name='conv1_2', input=conv1_2_input, group=1, conv_type='layers.Conv2D', filters=64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
        relu1_2         = layers.Activation(name='relu1_2', activation='relu')(conv1_2)
        pool1_input     = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)))(relu1_2)
        pool1           = layers.MaxPooling2D(name = 'pool1', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(pool1_input)
        conv2_1_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(pool1)
        conv2_1         = convolution(weights_dict, name='conv2_1', input=conv2_1_input, group=1, conv_type='layers.Conv2D', filters=128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
        relu2_1         = layers.Activation(name='relu2_1', activation='relu')(conv2_1)
        conv2_2_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu2_1)
        conv2_2         = convolution(weights_dict, name='conv2_2', input=conv2_2_input, group=1, conv_type='layers.Conv2D', filters=128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
        relu2_2         = layers.Activation(name='relu2_2', activation='relu')(conv2_2)
        pool2_input     = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)))(relu2_2)
        pool2           = layers.MaxPooling2D(name = 'pool2', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(pool2_input)
        conv3_1_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(pool2)
        conv3_1         = convolution(weights_dict, name='conv3_1', input=conv3_1_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
        relu3_1         = layers.Activation(name='relu3_1', activation='relu')(conv3_1)
        conv3_2_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu3_1)
        conv3_2         = convolution(weights_dict, name='conv3_2', input=conv3_2_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
        relu3_2         = layers.Activation(name='relu3_2', activation='relu')(conv3_2)
        conv3_3_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu3_2)
        conv3_3         = convolution(weights_dict, name='conv3_3', input=conv3_3_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
        relu3_3         = layers.Activation(name='relu3_3', activation='relu')(conv3_3)
        pool3_input     = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)))(relu3_3)
        pool3           = layers.MaxPooling2D(name = 'pool3', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(pool3_input)
        conv4_1_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(pool3)
        conv4_1         = convolution(weights_dict, name='conv4_1', input=conv4_1_input, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
        scale_pool3     = layers.Lambda(lambda x: x * 0.0001)(pool3)
        relu4_1         = layers.Activation(name='relu4_1', activation='relu')(conv4_1)
        score_pool3     = convolution(weights_dict, name='score_pool3', input=scale_pool3, group=1, conv_type='layers.Conv2D', filters=21, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
        conv4_2_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu4_1)
        conv4_2         = convolution(weights_dict, name='conv4_2', input=conv4_2_input, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
        relu4_2         = layers.Activation(name='relu4_2', activation='relu')(conv4_2)
        conv4_3_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu4_2)
        conv4_3         = convolution(weights_dict, name='conv4_3', input=conv4_3_input, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
        relu4_3         = layers.Activation(name='relu4_3', activation='relu')(conv4_3)
        pool4_input     = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)))(relu4_3)
        pool4           = layers.MaxPooling2D(name = 'pool4', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(pool4_input)
        conv5_1_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(pool4)
        conv5_1         = convolution(weights_dict, name='conv5_1', input=conv5_1_input, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
        scale_pool4     = layers.Lambda(lambda x: x * 0.01)(pool4)
        relu5_1         = layers.Activation(name='relu5_1', activation='relu')(conv5_1)
        score_pool4     = convolution(weights_dict, name='score_pool4', input=scale_pool4, group=1, conv_type='layers.Conv2D', filters=21, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
        conv5_2_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu5_1)
        conv5_2         = convolution(weights_dict, name='conv5_2', input=conv5_2_input, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
        relu5_2         = layers.Activation(name='relu5_2', activation='relu')(conv5_2)
        conv5_3_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu5_2)
        conv5_3         = convolution(weights_dict, name='conv5_3', input=conv5_3_input, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
        relu5_3         = layers.Activation(name='relu5_3', activation='relu')(conv5_3)
        pool5_input     = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)))(relu5_3)
        pool5           = layers.MaxPooling2D(name = 'pool5', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(pool5_input)
        fc6             = convolution(weights_dict, name='fc6', input=pool5, group=1, conv_type='layers.Conv2D', filters=4096, kernel_size=(7, 7), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
        relu6           = layers.Activation(name='relu6', activation='relu')(fc6)
        drop6           = layers.Dropout(name = 'drop6', rate = 0.5, seed = None)(relu6)
        fc7             = convolution(weights_dict, name='fc7', input=drop6, group=1, conv_type='layers.Conv2D', filters=4096, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
        relu7           = layers.Activation(name='relu7', activation='relu')(fc7)
        drop7           = layers.Dropout(name = 'drop7', rate = 0.5, seed = None)(relu7)
        score_fr        = convolution(weights_dict, name='score_fr', input=drop7, group=1, conv_type='layers.Conv2D', filters=21, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=True)
        upscore2        = convolution(weights_dict, name='upscore2', input=score_fr, group=1, conv_type='layers.Conv2DTranspose', filters=21, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(1, 1), padding='valid', use_bias=False)
        score_pool4c    = layers.Cropping2D(cropping=((5, 5), (5, 5)), name='score_pool4c')(score_pool4)
        fuse_pool4      = layers.add(name = 'fuse_pool4', inputs = [upscore2, score_pool4c])
        upscore_pool4   = convolution(weights_dict, name='upscore_pool4', input=fuse_pool4, group=1, conv_type='layers.Conv2DTranspose', filters=21, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(1, 1), padding='valid', use_bias=False)
        score_pool3c    = layers.Cropping2D(cropping=((9, 9), (9, 9)), name='score_pool3c')(score_pool3)
        fuse_pool3      = layers.add(name = 'fuse_pool3', inputs = [upscore_pool4, score_pool3c])
        upscore8        = convolution(weights_dict, name='upscore8', input=fuse_pool3, group=1, conv_type='layers.Conv2DTranspose', filters=21, kernel_size=(16, 16), strides=(8, 8), dilation_rate=(1, 1), padding='valid', use_bias=False)
        score           = layers.Cropping2D(cropping=((31, 31), (31, 31)), name='score')(upscore8)
        model           = Model(inputs = [input], outputs = [score])
        
        set_layer_weights(model, weights_dict)
        return model
        
    def convolution(weights_dict, name, input, group, conv_type, filters=None, **kwargs):
        if not conv_type.startswith('layer'):
            layer = keras.applications.mobilenet.DepthwiseConv2D(name=name,
                                                                 **kwargs)(input)
            return layer

        grouped_channels = int(filters / group)
        group_list = []

        if group == 1:
            func = getattr(layers, conv_type.split('.')[-1])
            layer = func(name = name, filters = filters, **kwargs)(input)
            return layer

        weight_groups = list()
        if not weights_dict == None:
            w = numpy.array(weights_dict[name]['weights'])
            weight_groups = numpy.split(w, indices_or_sections=group, axis=-1)

        for c in range(group):
            x = layers.Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(input)
            x = layers.Conv2D(name=name + "_" + str(c), filters=grouped_channels, **kwargs)(x)
            weights_dict[name + "_" + str(c)] = dict()
            weights_dict[name + "_" + str(c)]['weights'] = weight_groups[c]

            group_list.append(x)

        layer = layers.concatenate(group_list, axis = -1)

        if 'bias' in weights_dict[name]:
            b = K.variable(weights_dict[name]['bias'], name = name + "_bias")
            layer = layer + b
        return layer

    
if __name__ == '__main__':
    Mask().run()