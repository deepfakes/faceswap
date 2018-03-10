import cv2
import numpy

from lib.umeyama import umeyama

def random_transform( image, rotation_range, zoom_range, shift_range, random_flip ):
    h,w = image.shape[0:2]
    rotation = numpy.random.uniform( -rotation_range, rotation_range )
    scale = numpy.random.uniform( 1 - zoom_range, 1 + zoom_range )
    tx = numpy.random.uniform( -shift_range, shift_range ) * w
    ty = numpy.random.uniform( -shift_range, shift_range ) * h
    mat = cv2.getRotationMatrix2D( (w//2,h//2), rotation, scale )
    mat[:,2] += (tx,ty)
    result = cv2.warpAffine( image, mat, (w,h), borderMode=cv2.BORDER_REPLICATE )
    if numpy.random.random() < random_flip:
        result = result[:,::-1]
    return result

maskColour = numpy.array([127,127,127])/255.0


from scipy.interpolate import griddata


grid_x, grid_y = numpy.mgrid[0:255:256j, 0:255:256j]
edgeAnchors = [ (0,0), (0,255), (255,255), (255,0), (127,0), (127,255), (255,127), (0,127) ]

def random_warp_src_dest( in_image,srcPoints,destPoints ):

  source = srcPoints
  destination = (destPoints.copy().astype('float')+numpy.random.normal( size=(destPoints.shape), scale= 2 ))
  destination = destination.astype('uint8')

  faceCore =cv2.convexHull( numpy.concatenate([source[17:],destination[17:]], axis=0).astype(int) )

  source = [(y,x) for x,y in source]+edgeAnchors
  destination = [(y,x) for x,y in destination]+edgeAnchors


  indeciesToRemove = set()
  for fpl in source,destination:
    for i,(y,x) in enumerate(fpl):
      if i>17:
        break
      elif cv2.pointPolygonTest(faceCore,(y,x),False) >= 0:
        indeciesToRemove.add(i)

  for i in sorted(indeciesToRemove,reverse=True):
    source.pop(i)
    destination.pop(i)

  grid_z = griddata( destination,source, (grid_x, grid_y), method='linear')
  map_x = numpy.append([], [ar[:,1] for ar in grid_z]).reshape(256,256)
  map_y = numpy.append([], [ar[:,0] for ar in grid_z]).reshape(256,256)
  map_x_32 = map_x.astype('float32')
  map_y_32 = map_y.astype('float32')

  warped = cv2.remap(in_image[:,:,:3], map_x_32, map_y_32, cv2.INTER_LINEAR,cv2.BORDER_TRANSPARENT )

  target_mask = in_image[:,:,3].reshape((256,256,1))
  target_image = in_image[:,:,:3]

  warped       = cv2.resize( warped[ 128-120:128+120,128-120:128+120,:]         ,(64,64),cv2.INTER_AREA)
  target_image = cv2.resize( target_image[ 128-120:128+120,128-120:128+120,: ]  ,(64*2,64*2),cv2.INTER_AREA)
  target_mask  = cv2.resize( target_mask[ 128-120:128+120,128-120:128+120,: ]   ,(64*2,64*2),cv2.INTER_AREA).reshape((64*2,64*2,1))

  return warped,target_image,target_mask

# # get pair of random warped images from aligened face image
# def random_warp( in_image ):
#     assert in_image.shape[:2] == (256,256)

#     image = in_image.copy()


#     scale = 5

#     range_ = numpy.linspace( 128-120, 128+120, scale )
#     mapx = numpy.broadcast_to( range_, (scale,scale) )
#     mapy = mapx.T

#     mapx = mapx + numpy.random.normal( size=(scale,scale), scale= 6 )
#     mapy = mapy + numpy.random.normal( size=(scale,scale), scale= 6 )

#     interp_mapx = cv2.resize( mapx, (80,80) )[8:72,8:72].astype('float32')
#     interp_mapy = cv2.resize( mapy, (80,80) )[8:72,8:72].astype('float32')

#     warped_image = cv2.remap( image[:,:,:3], interp_mapx, interp_mapy, cv2.INTER_CUBIC )

#     src_points = numpy.stack( [ mapx.ravel(), mapy.ravel() ], axis=-1 )
#     dst_points = numpy.mgrid[0:65:16,0:65:16].T.reshape(-1,2)
#     mat = umeyama( src_points, dst_points, True )[0:2]

#     target_image = cv2.warpAffine( image, mat, (64,64) )

#     target_mask = target_image[:,:,3].reshape((64,64,1))
#     target_image = target_image[:,:,:3]


#     if len(target_image.shape)>2:
#       return ( warped_image, 
#                target_image, 
#                target_mask )
#     else:
#       return ( warped_image, 
#                target_image )