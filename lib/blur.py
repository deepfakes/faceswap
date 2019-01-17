

def fft_convolve2d(image, kernel):
    """ 2D convolution, using FFT"""
	
	def pad_to_power(arr, kernel):
		next_power = NextPowerOfTwo(numpy.max(arr.shape))
		next_size = numpy.power(2, next_power)
		y_deficit, x_deficit, _ = next_size - arr.shape
		a_deficit, b_deficit, _ = next_size - arr.shape
		image = numpy.pad(arr, ((y_deficit,0),(x_deficit,0)), mode='constant')
		kernel = numpy.pad(arr, ((a_deficit,0),(b_deficit,0)), mode='constant')
		
		return image, kernel
	
	padded_image, padded_kernel = pad_to_power(image, kernel)
    fr = numpy.fft.fft2(padded_image)
    fr2 = numpy.fft.fft2(numpy.flipud(numpy.fliplr(padded_kernel)))
    m,n = fr.shape
    cc = numpy.real(numpy.fft.ifft2(fr * fr2))
    cc = numpy.roll(cc, -m/2+1, axis=0)
    cc = numpy.roll(cc, -n/2+1, axis=1)
	
    return cc

