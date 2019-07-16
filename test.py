import os
import cv2
import imageio
import imageio_ffmpeg as im_ffm

out = "C:\\Users\\matt\\Desktop\\out"
vidname = "test"
reader = imageio.get_reader("C:\\Users\\matt\\Desktop\\2.mkv")
for i, frame in enumerate(reader):
    #Convert to BGR for cv2 compatibility
    frame = frame[:, :, ::-1]
    if frame is None or not frame.any():
        print(frame.shape)
        print(frame.ndim)
            
    filename = "{}_{:06d}.png".format(vidname, i + 1)
    outfile = os.path.join(out, filename)
    print(filename)
    break
    cv2.imwrite(outfile, frame)
    #logger.trace("Loading video frame: '%s'", filename)
reader.close()
