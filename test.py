import os
import cv2
import numpy as np
from lib.logger import log_setup
log_setup("INFO", None, "test", False)

from lib.alignments import Alignments
from lib.faces_detect import DetectedFace

path = "/home/matt/fake/test/extract/sb_alignments.fsa"
#path = "/home/matt/fake/test/extract/src/alignments.fsa"

t_file = "/home/matt/fake/test/extract/sb_test/sb_000040_0.png"
from hashlib import sha1
img = cv2.imread(t_file)

print(sha1(img).hexdigest())

f_folder = "/home/matt/fake/test/extract/srcout"
a = Alignments(os.path.dirname(path), filename=os.path.basename(path))

print(a.data["sb_000040.png"][0]["hash"])
exit(0)

for frame, faces in a.data.items():
    if not faces:
        continue
    print(frame, [face["mask"]["extended"]["frame_dims"] for face in faces])
    continue
    fname, ext = os.path.splitext(frame)
    for idx, face in enumerate(faces):
        detface = DetectedFace()
        detface.from_alignment(face)
        imgpath = os.path.join(f_folder, "{}_{}{}".format(fname, idx, ext))
        img = cv2.imread(imgpath)
        outfname, ext = os.path.splitext(os.path.basename(imgpath))
        outiname = "{}_orig{}".format(outfname, ext)
        outipath = os.path.join("/home/matt/fake/test/testing/", outiname)
        cv2.imwrite(outipath, img)
        for masktype, mask in detface.mask.items():
            outmname = "{}_{}{}".format(outfname, masktype, ext)
            outpath = os.path.join("/home/matt/fake/test/testing/", outmname)
            mask.set_blur_kernel_and_threshold(blur_kernel=7, threshold=20)
            face_mask = mask.mask
            outimg = np.concatenate((img, cv2.resize(face_mask, (256, 256), interpolation=cv2.INTER_CUBIC)[..., None]), axis=-1)
            cv2.imwrite(outpath, outimg)


