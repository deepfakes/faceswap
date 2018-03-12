import cv2
import numpy
import os
import json

from tqdm import tqdm

#guideShades = numpy.linspace(20,250,68)

def load_images_aligned(image_paths):
  basePath = os.path.split(image_paths[0])[0]
  alignments = os.path.join(basePath,'alignments.json')
  alignments = json.loads( open(alignments).read() )

  all_images    = []
  landmarks     = []


  pbar = tqdm(alignments)
  for original,cropped,mat,points in pbar:
    pbar.set_description('loading '+basePath)
    cropped = os.path.split(cropped)[1]
    cropped = os.path.join(basePath,cropped)
    if cropped in image_paths and os.path.exists(cropped):
        image, facepoints = get_image( cropped, mat, points )
        all_images.append( image  )
        landmarks.append( facepoints )

  return numpy.array(all_images),numpy.array(landmarks)

def load_image(cropped, mat, points):
    cropped = cv2.imread(cropped).astype(float)

    mat = numpy.array(mat).reshape(2,3)
    points = numpy.array(points).reshape((-1,2))

    mat = mat*160
    mat[:,2] += 42

    facepoints = numpy.array( points ).reshape((-1,2))

    mask = numpy.zeros_like(cropped,dtype=numpy.uint8)

    hull = cv2.convexHull( facepoints.astype(int) )
    hull = cv2.transform( hull.reshape(1,-1,2) , mat).reshape(-1,2).astype(int)

    cv2.fillConvexPoly( mask,hull,(255,255,255) )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

    mask = cv2.dilate(mask,kernel,iterations = 1,borderType=cv2.BORDER_REFLECT )

    facepoints = cv2.transform( numpy.array( points ).reshape((1,-1,2)) , mat).reshape(-1,2).astype(int)

    mask = mask[:,:,0]

    return numpy.dstack([cropped,mask]).astype(numpy.uint8), facepoints
