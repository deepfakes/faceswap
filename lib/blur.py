
import numpy, cv2, timeit

class Mask():
    def __init__(self):
   
        def fft_convolve2d(self, image, size=21, sigma=5.0):
            ''' 2D convolution, using FFT'''
            
            def gaussian_kernel(size, sigma):
                kernelX = cv2.getGaussianKernel(size, sigma)
                kernelY = cv2.getGaussianKernel(size, sigma)
                kernelXY = kernelX * kernelY.T
                return kernelXY
                
            def pad_to_power(arr, kernel):
                next_power = numpy.ceil(numpy.log2(numpy.max(arr.shape)))
                next_size = int(numpy.power(2, next_power))
                y_deficit, x_deficit = next_size - arr.shape[0], next_size - arr.shape[1]
                a_deficit, b_deficit = next_size - arr.shape[0], next_size - arr.shape[1]
                arr = numpy.squeeze(arr)
                image = numpy.pad(arr, ((y_deficit,0),(x_deficit,0)), mode='constant')
                kernel = numpy.pad(arr, ((a_deficit,0),(b_deficit,0)), mode='constant')
                
                return image, kernel
            
            kernel = gaussian_kernel(size, sigma)
            padded_image, padded_kernel = pad_to_power(image, kernel)
            fr = numpy.fft.fft2(padded_image)
            fr2 = numpy.fft.fft2(numpy.flipud(numpy.fliplr(padded_kernel)))
            m,n = fr.shape
            cc = numpy.real(numpy.fft.ifft2(fr * fr2))
            cc = numpy.roll(cc, -m/2+1, axis=0)
            cc = numpy.roll(cc, -n/2+1, axis=1)
            
            return cc
            
        mysetup = '''
import numpy, cv2

def gaussian_kernel_1(k=21, sigma=5.0):
    probs = [numpy.exp(-z*z/(2*sigma*sigma))/numpy.sqrt(2*numpy.pi*sigma*sigma) for z in range(-k//2,k//2+1)] 
    kernel = numpy.outer(probs, probs)
    return kernel

def gaussian_kernel_2(size=21, sigma=5.0):
    x, y = numpy.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = numpy.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def gaussian_kernel_3(size=21, sigma=5.0):
    ax = numpy.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = numpy.meshgrid(ax, ax, copy=False)
    kernel = numpy.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / numpy.sum(kernel)

def gaussian_kernel_4(shape=(21,21),sigma=5.0):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = numpy.ogrid[-m:m+1,-n:n+1]
    h = numpy.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < numpy.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
    
def gaussian_kernel_5(size=21, sigma=5.0):
    kernelX = cv2.getGaussianKernel(size, sigma)
    kernelY = cv2.getGaussianKernel(size, sigma)
    kernelXY = kernelX * kernelY.T
    return kernelXY
'''

        mysetup = '''
import numpy

def gaussian_kernel_1(size=21, sigma=5.0):
    xx, yy = numpy.mgrid[-size // 2 + 1: size // 2 + 1, -size // 2 + 1: size // 2 + 1]

def gaussian_kernel_2(size=21, sigma=5.0):
    xx, yy = numpy.mgrid[-size // 2 + 1: size // 2 + 1, -size // 2 + 1: size // 2 + 1]
    kernel = numpy.exp(-(xx**2 + yy**2) / (2. * sigma**2))

def gaussian_kernel_3(size=21, sigma=5.0):
    xx, yy = numpy.mgrid[-size // 2 + 1: size // 2 + 1, -size // 2 + 1: size // 2 + 1]
    kernel = numpy.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / numpy.sum(kernel)
'''
        
        a = timeit.repeat(setup = mysetup, stmt = 'x = gaussian_kernel_1()', repeat = 3, number = 10000)
        b = timeit.repeat(setup = mysetup, stmt = 'x = gaussian_kernel_2()', repeat = 3, number = 10000) 
        c = timeit.repeat(setup = mysetup, stmt = 'x = gaussian_kernel_3()', repeat = 3, number = 10000) 
        #d = timeit.repeat(setup = mysetup, stmt = 'x = gaussian_kernel_4()', repeat = 3, number = 10000) 
        #e = timeit.repeat(setup = mysetup, stmt = 'x = gaussian_kernel_5()', repeat = 3, number = 10000)   
        
        print('a time: {}'.format(min(a)))
        print('b time: {}'.format(min(b)))
        print('c time: {}'.format(min(c)))
        #print('d time: {}'.format(min(d)))
        #print('e time: {}'.format(min(e)))
        

        
if __name__ == '__main__':
    Mask()
