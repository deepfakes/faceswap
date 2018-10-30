import keras.backend as K
import tensorflow as tf


class DSSIMObjective():
    def __init__(self, k1=0.01, k2=0.03, kernel_size=3, max_value=1.0):
        """
        Difference of Structural Similarity (DSSIM loss function). Clipped between 0 and 0.5
        Note : You should add a regularization term like a l2 loss in addition to this one.
        Note : In theano, the `kernel_size` must be a factor of the output size. So 3 could
               not be the `kernel_size` for an output of 32.
        
        # Arguments
            k1: Parameter of the SSIM (default 0.01)
            k2: Parameter of the SSIM (default 0.03)
            kernel_size: Size of the sliding window (default 3)
            max_value: Max value of the output (default 1.0)
        """
        self.__name__ = 'DSSIMObjective'
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value) ** 2
        self.c2 = (self.k2 * self.max_value) ** 2
        self.dim_ordering = K.image_data_format()
        self.backend = K.backend()
        
    @staticmethod
    def permute_dimensions(x, pattern):
        """Permutes axes in a tensor.
        
        # Arguments
            x: Tensor or variable.
            pattern: A tuple of
                dimension indices, e.g. `(0, 2, 1)`.
        
        # Returns
            A tensor.
        """
        return tf.transpose(x, perm=pattern)
    
    @staticmethod
    def _preprocess_padding(padding):
        """Convert keras' padding to tensorflow's padding.
        # Arguments
            padding: string, `"same"` or `"valid"`.
        # Returns
            a string, `"SAME"` or `"VALID"`.
        # Raises
            ValueError: if `padding` is invalid.
        """
        if padding == 'same':
            padding = 'SAME'
        elif padding == 'valid':
            padding = 'VALID'
        else:
            raise ValueError('Invalid padding:', padding)
        return padding
    
    @staticmethod
    def int_shape(x):
        """Returns the shape of tensor or variable as a tuple of int or None entries.
        
        # Arguments
            x: Tensor or variable.
        
        # Returns
            A tuple of integers (or None entries).
        
        # Examples
        ```python
            >>> from keras import backend as K
            >>> inputs = K.placeholder(shape=(2, 4, 5))
            >>> K.int_shape(inputs)
            (2, 4, 5)
            >>> val = np.array([[1, 2], [3, 4]])
            >>> kvar = K.variable(value=val)
            >>> K.int_shape(kvar)
            (2, 2)
        ```
        """
        if hasattr(x, '_keras_shape'):
            return x._keras_shape
        try:
            return tuple(x.get_shape().as_list())
        except ValueError:
            return None            
    
    @classmethod
    def extract_image_patches(cls, x, ksizes, ssizes, padding='same',
              data_format='channels_last'):
        '''
        Extract the patches from an image
        # Parameters
        
            x : The input image
            ksizes : 2-d tuple with the kernel size
            ssizes : 2-d tuple with the strides size
            padding : 'same' or 'valid'
            data_format : 'channels_last' or 'channels_first'
        
        # Returns
            The (k_w,k_h) patches extracted
            TF ==> (batch_size,w,h,k_w,k_h,c)
            TH ==> (batch_size,w,h,c,k_w,k_h)
        '''
        kernel = [1, ksizes[0], ksizes[1], 1]
        strides = [1, ssizes[0], ssizes[1], 1]
        padding = cls._preprocess_padding(padding)
        if data_format == 'channels_first':
            x = self.permute_dimensions(x, (0, 2, 3, 1))
        bs_i, w_i, h_i, ch_i = cls.int_shape(x)
        patches = tf.extract_image_patches(x, kernel, strides, [1, 1, 1, 1],
                                           padding)
        # Reshaping to fit Theano
        bs, w, h, ch = cls.int_shape(patches)
        patches = tf.reshape(tf.transpose(tf.reshape(patches, [-1, w, h, tf.floordiv(ch, ch_i), ch_i]), [0, 1, 2, 4, 3]),
                             [-1, w, h, ch_i, ksizes[0], ksizes[1]])
        if data_format == 'channels_last':
            patches = cls.permute_dimensions(patches, [0, 1, 2, 4, 5, 3])
        return patches
        
    def __int_shape(self, x):
        return self.int_shape(x) # K.int_shape(x) if self.backend == 'tensorflow' else K.shape(x)
        
    def __call__(self, y_true, y_pred):
        # There are additional parameters for this function
        # Note: some of the 'modes' for edge behavior do not yet have a gradient definition in the Theano tree
        #   and cannot be used for learning
        
        kernel = [self.kernel_size, self.kernel_size]
        y_true = tf.reshape(y_true, [-1] + list(self.__int_shape(y_pred)[1:]))
        y_pred = tf.reshape(y_pred, [-1] + list(self.__int_shape(y_pred)[1:]))
        
        patches_pred = self.extract_image_patches(y_pred, kernel, kernel, 'valid', self.dim_ordering)
        patches_true = self.extract_image_patches(y_true, kernel, kernel, 'valid', self.dim_ordering)
        
        # Reshape to get the var in the cells
        bs, w, h, c1, c2, c3 = self.__int_shape(patches_pred)
        patches_pred = tf.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
        patches_true = tf.reshape(patches_true, [-1, w, h, c1 * c2 * c3])
        # Get mean
        u_true = K.mean(patches_true, axis=-1)
        u_pred = K.mean(patches_pred, axis=-1)
        # Get variance
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        # Get std dev
        covar_true_pred = K.mean(patches_true * patches_pred, axis=-1) - u_true * u_pred
        
        ssim = (2 * u_true * u_pred + self.c1) * (2 * covar_true_pred + self.c2)
        denom = (K.square(u_true) + K.square(u_pred) + self.c1) * (var_pred + var_true + self.c2)
        ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
        return K.mean((1.0 - ssim) / 2.0)
