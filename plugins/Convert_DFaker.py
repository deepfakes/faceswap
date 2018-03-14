import argparse
import cv2
import json
import numpy
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
# from model import autoencoder_A
# from model import autoencoder_B
# from model import encoder, decoder_A, decoder_B

# encoder  .load_weights( "models/encoder.h5"   )
# decoder_A.load_weights( "models/decoder_A.h5" )
# decoder_B.load_weights( "models/decoder_B.h5" )
import time

n=0

imageSize = 256
croppedSize = 240 
zmask = numpy.zeros((1,128, 128,1),float)

import cv2
import numpy

from lib.aligner import get_align_mat

class Convert():
    def __init__(self, encoder, trainer, blur_size=2, seamless_clone=False, mask_type="facehullandrect", erosion_kernel_size=None, match_histogram=False, sharpen_image=None, double_pass=None, **kwargs):
        self.encoder_A = encoder(False)
        self.encoder_B = encoder(True)
        self.trainer = trainer
        self.erosion_kernel = None
        self.erosion_kernel_size = erosion_kernel_size
        if erosion_kernel_size is not None:
            self.erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(abs(erosion_kernel_size),abs(erosion_kernel_size)))
        self.blur_size = blur_size
        self.seamless_clone = seamless_clone
        self.sharpen_image = sharpen_image
        self.match_histogram = match_histogram
        self.mask_type = mask_type.lower() # Choose in 'FaceHullAndRect','FaceHull','Rect'
        self.double_pass = double_pass

    def patch_image( self, image, face_detected, size ):
        mat = numpy.array(get_align_mat(face_detected, size, should_align_eyes=False)).reshape(2,3)

        global n
        n+=1
        size = 64
        image_size = image.shape[1], image.shape[0]

        sourceMat = mat.copy()

        sourceMat = sourceMat*(240+(16*2))
        sourceMat[:,2] += 48

        face = cv2.warpAffine( image, sourceMat, (240+(48+16)*2,240+(48+16)*2) )
        
        sourceFace = face.copy()
        sourceFace = cv2.resize(sourceFace,(128,128),cv2.INTER_CUBIC)
        

        face = cv2.resize(face,(64,64),cv2.INTER_AREA)
        face = numpy.expand_dims( face, 0 )

        new_face_rgb,new_face_m = self.encoder_A( [face / 255.0,zmask] )

        if self.double_pass:
            #feed the original prediction back into the network for a second round.
            new_face_rgb = new_face_rgb.reshape((128, 128, 3))
            new_face_rgb = cv2.resize( new_face_rgb , (64,64))
            new_face_rgb = numpy.expand_dims( new_face_rgb, 0 )
            new_face_rgb,_ = self.encoder_B( [new_face_rgb,zmask] )
        

        _,other_face_m = self.encoder_A( [face / 255.0,zmask] ) # NOTE: we get the image from other autoencoder!

        new_face_m = numpy.maximum(new_face_m, other_face_m )


        new_face_rgb = numpy.clip( new_face_rgb[0] * 255, 0, 255 ).astype( image.dtype )
        new_face_m   = numpy.clip( new_face_m[0]  , 0, 1 ).astype( float ) * numpy.ones((new_face_m.shape[0],new_face_m.shape[1],3))


        base_image = numpy.copy( image )
        new_image = numpy.copy( image ) 

        transmat =  mat * (64-16) *2
        transmat[::,2] += 8*2

        self.adjust_avg_color(sourceFace,new_face_rgb)

        cv2.warpAffine( new_face_rgb, transmat, image_size, new_image, cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT )
        

        image_mask = numpy.zeros_like(new_image, dtype=float)


        cv2.warpAffine( new_face_m, transmat, image_size, image_mask, cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT )
        

        #slightly enlarge the mask area
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        image_mask = cv2.dilate(image_mask,kernel,iterations = 1)

        if self.seamless_clone:

            unitMask = numpy.clip( image_mask * 365, 0, 255 ).astype(numpy.uint8)

            maxregion = numpy.argwhere(unitMask==255)

            if maxregion.size > 0:

                miny,minx = maxregion.min(axis=0)[:2]
                maxy,maxx = maxregion.max(axis=0)[:2]
                lenx = maxx - minx;
                leny = maxy - miny;
                masky = int(minx+(lenx//2))
                maskx = int(miny+(leny//2))

                new_image = cv2.seamlessClone(new_image.astype(numpy.uint8),base_image.astype(numpy.uint8),unitMask,(masky,maskx) , cv2.NORMAL_CLONE )


        image_mask = cv2.GaussianBlur(image_mask,(11,11),0)

        foreground = cv2.multiply(image_mask, new_image.astype(float))
        background = cv2.multiply(1.0 - image_mask, base_image.astype(float))
        
        output = numpy.add(background,foreground)

        # cv2.imshow("output", output.astype(numpy.uint8) )

        # if cv2.waitKey(1)==ord('q'):
        #   exit()
        return output

    def adjust_avg_color(self, img_old,img_new):
        w,h,c = img_new.shape
        for i in range(img_new.shape[-1]):
            old_avg = img_old[:, :, i].mean()
            new_avg = img_new[:, :, i].mean()
            diff_int = (int)(old_avg - new_avg)
            for m in range(img_new.shape[0]):
                for n in range(img_new.shape[1]):
                    temp = (img_new[m,n,i] + diff_int)
                    if temp < 0:
                        img_new[m,n,i] = 0
                    elif temp > 255:
                        img_new[m,n,i] = 255
                    else:
                        img_new[m,n,i] = temp

    # def transfer_avg_color(self, img_old,img_new):
    #     assert(img_old.shape==img_new.shape) 
    #     source = cv2.cvtColor(img_old, cv2.COLOR_BGR2LAB).astype("float32")
    #     target = cv2.cvtColor(img_new, cv2.COLOR_BGR2LAB).astype("float32")

    #     (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = self.image_stats(source)
    #     (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = self.image_stats(target)

    #     (l, a, b) = cv2.split(target)

    #     l -= lMeanTar
    #     a -= aMeanTar
    #     b -= bMeanTar

    #     l = (lStdTar / lStdSrc) * l
    #     a = (aStdTar / aStdSrc) * a
    #     b = (bStdTar / bStdSrc) * b

    #     l += lMeanSrc
    #     a += aMeanSrc
    #     b += bMeanSrc

    #     l = numpy.clip(l, 0, 255)
    #     a = numpy.clip(a, 0, 255)
    #     b = numpy.clip(b, 0, 255)

    #     transfer = cv2.merge([l, a, b])
    #     transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    #     return transfer

    # def image_stats(self, image):
    #     (l, a, b) = cv2.split(image)
    #     (lMean, lStd) = (l.mean(), l.std())
    #     (aMean, aStd) = (a.mean(), a.std())
    #     (bMean, bStd) = (b.mean(), b.std())

    #     return (lMean, lStd, aMean, aStd, bMean, bStd)

# def main( args ):


#     input_dir = Path( args.input_dir )
#     assert input_dir.is_dir()

#     alignments = input_dir / args.alignments
#     with alignments.open() as f:
#         alignments = json.load(f)

#     output_dir = input_dir / args.output_dir
#     output_dir.mkdir( parents=True, exist_ok=True )

#     args.direction = 'AtoB'
#     if args.direction == 'AtoB': autoencoder,otherautoencoder = autoencoder_B,autoencoder_A
#     if args.direction == 'BtoA': autoencoder,otherautoencoder = autoencoder_A,autoencoder_B
    
#     if args.blurSize % 2 == 0:
#       args.blurSize+=1

#     if args.erosionKernelSize>0:
#       erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(args.erosionKernelSize,args.erosionKernelSize))
#     else:
#       erosion_kernel = None

#     for e in alignments:    
#       if len(e)<4:
#         raise LookupError('This script expects new format json files with face points included.')


#     for image_file, face_file, mat,facepoints in tqdm( alignments[args.startframe::args.frameSkip] ):
#         image = cv2.imread( str( input_dir / image_file ) )
#         face  = cv2.imread( str( input_dir / face_file  ) )


#         mat = numpy.array(mat).reshape(2,3)

#         if image is None: continue
#         if face  is None: continue


#         new_image = convert_one_image( autoencoder, otherautoencoder, image, mat, facepoints, erosion_kernel, args.blurSize, args.seamlessClone, args.maskType, args.doublePass)

#         output_file = output_dir / Path(image_file).name
#         cv2.imwrite( str(output_file), new_image )

# def str2bool(v):
#   if v.lower() in ('yes', 'true', 't', 'y', '1'):
#     return True
#   elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#     return False
#   else:
#     raise argparse.ArgumentTypeError('Boolean value expected.')

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument( "input_dir", type=str, nargs='?' )
#     parser.add_argument( "alignments", type=str, nargs='?', default='alignments.json' )
#     parser.add_argument( "output_dir", type=str, nargs='?', default='merged' )

#     parser.add_argument("--seamlessClone", type=str2bool, nargs='?', const=False, default='False', help="Attempt to use opencv seamlessClone.")

#     parser.add_argument("--doublePass", type=str2bool, nargs='?', const=False, default='False', help="Pass the original prediction output back through for a second pass.")


#     parser.add_argument('--maskType', type=str, default='FaceHullAndRect' ,choices=['FaceHullAndRect','FaceHull','Rect'], help="The type of masking to use around the face.")

#     parser.add_argument( "--startframe",        type=int, default='0' )
#     parser.add_argument( "--frameSkip",        type=int, default='1' )
#     parser.add_argument( "--blurSize",          type=int, default='4' )
#     parser.add_argument( "--erosionKernelSize", type=int, default='2' )
#     parser.add_argument( "--direction",         type=str, default="AtoB", choices=["AtoB", "BtoA"])
#     main( parser.parse_args() )

