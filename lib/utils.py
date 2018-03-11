import cv2
import sys
from os.path import basename, exists

from pathlib import Path
from scandir import scandir

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

    dir_scanned = list(scandir(directory))
    for x in dir_scanned:
        if any([x.name.lower().endswith(ext) for ext in image_extensions]):
            if x.name in exclude_names:
                if debug:
                    print("Already processed %s" % x.name)
                continue
            else:
                dir_contents.append(x.path)

    return dir_contents

def rotate_image(image, angle):
    ''' Rotates an image by 90, 180 or 270 degrees. Positive for clockwise, negative for 
        counterclockwise '''
    if angle < 0: angle = sum((360, angle))
    if angle == 90:
        image = cv2.flip(cv2.transpose(image),flipCode=1)
    elif angle == 180:
        image = cv2.flip(image,flipCode=-1)
    elif angle == 270:
        image = cv2.flip(cv2.transpose(image),flipCode=0)
    else:
        print('Unsupported image rotation angle: {}. Image unmodified'.format(angle))
    return image

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
