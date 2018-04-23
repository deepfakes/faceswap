import cv2
import sys
from os.path import basename, exists, join

from pathlib import Path
from scandir import scandir
import os

image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

def get_folder(path):
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_image_paths(directory, exclude=[], debug=False):
    exclude_names = [basename(Path(x).stem[:-1] + Path(x).suffix) for x in exclude]
    dir_contents = []

    if not exists(directory):
        directory = get_folder(directory).path

    dir_scanned = sorted(os.scandir(directory), key=lambda x: x.name)
    for x in dir_scanned:
        if any([x.name.lower().endswith(ext) for ext in image_extensions]):
            if x.name in exclude_names:
                if debug:
                    print("Already processed %s" % x.name)
                continue
            else:
                dir_contents.append(x.path)

    return dir_contents

def backup_file(directory, filename):
    """ Backup a given file by appending .bk to the end """
    origfile = join(directory, filename)
    backupfile = origfile + '.bk'
    if exists(backupfile):
        os.remove(backupfile)
    os.rename(origfile, backupfile)

# From: https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
def rotate_image(image, angle, rotated_width=None, rotated_height=None):
    height, width = image.shape[:2]
    image_center = (width/2, height/2)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, -1.*angle, 1.)
    if rotated_width is None or rotated_height is None:
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        if rotated_width is None:
            rotated_width = int(height*abs_sin + width*abs_cos)
        if rotated_height is None:
            rotated_height = int(height*abs_cos + width*abs_sin)
    rotation_matrix[0, 2] += rotated_width/2 - image_center[0]
    rotation_matrix[1, 2] += rotated_height/2 - image_center[1]
    return cv2.warpAffine(image, rotation_matrix, (rotated_width, rotated_height))

# From: https://stackoverflow.com/questions/7323664/python-generator-pre-fetch
import threading
import queue as Queue
class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, prefetch=1): #See below why prefetch count is flawed
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        # Put until queue size is reached. Note: put blocks only if put is called while queue has already reached max size
        # => this makes 2 prefetched items! One in the queue, one waiting for insertion!
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def iterator(self):
        while True:
            next_item = self.queue.get()
            if next_item is None:
                break
            yield next_item
