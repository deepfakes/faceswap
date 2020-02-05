from lib.logger import log_setup
log_setup("INFO", None, None, False)

from lib.image import ImagesLoader

a = ImagesLoader("/home/matt/fake/test/extract/match.mp4")

b = a.frame_from_index(10)
print([c for c in b])