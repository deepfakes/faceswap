#!/usr/bin python3
""" Face Filterer for extraction in faceswap.py """

import logging

from lib.vgg_face import VGGFace
from lib.image import read_image
from plugins.extract.pipeline import Extractor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def avg(arr):
    """ Return an average """
    return sum(arr) * 1.0 / len(arr)


class FaceFilter():
    """ Face filter for extraction
        NB: we take only first face, so the reference file should only contain one face. """

    def __init__(self, reference_file_paths, nreference_file_paths, detector, aligner,
                 multiprocess=False, threshold=0.4):
        logger.debug("Initializing %s: (reference_file_paths: %s, nreference_file_paths: %s, "
                     "detector: %s, aligner: %s, multiprocess: %s, threshold: %s)",
                     self.__class__.__name__, reference_file_paths, nreference_file_paths,
                     detector, aligner, multiprocess, threshold)
        self.vgg_face = VGGFace()
        self.filters = self.load_images(reference_file_paths, nreference_file_paths)
        # TODO Revert face-filter to use the selected detector and aligner.
        # Currently Tensorflow does not release vram after it has been allocated
        # Whilst this vram can still be used, the pipeline for the extraction process can't see
        # it so thinks there is not enough vram available.
        # Either the pipeline will need to be changed to be re-usable by face-filter and extraction
        # Or another vram measurement technique will need to be implemented to for when TF has
        # already performed allocation. For now we force CPU detectors.

        # self.align_faces(detector, aligner, multiprocess)
        self.align_faces("cv2-dnn", "cv2-dnn", "none", multiprocess)

        self.get_filter_encodings()
        self.threshold = threshold
        logger.debug("Initialized %s", self.__class__.__name__)

    @staticmethod
    def load_images(reference_file_paths, nreference_file_paths):
        """ Load the images """
        retval = dict()
        for fpath in reference_file_paths:
            retval[fpath] = {"image": read_image(fpath, raise_error=True),
                             "type": "filter"}
        for fpath in nreference_file_paths:
            retval[fpath] = {"image": read_image(fpath, raise_error=True),
                             "type": "nfilter"}
        logger.debug("Loaded filter images: %s", {k: v["type"] for k, v in retval.items()})
        return retval

    # Extraction pipeline
    def align_faces(self, detector_name, aligner_name, masker_name, multiprocess):
        """ Use the requested detectors to retrieve landmarks for filter images """
        extractor = Extractor(detector_name,
                              aligner_name,
                              masker_name,
                              multiprocess=multiprocess)
        self.run_extractor(extractor)
        del extractor
        self.load_aligned_face()

    def run_extractor(self, extractor):
        """ Run extractor to get faces """
        for _ in range(extractor.passes):
            self.queue_images(extractor)
            extractor.launch()
            for faces in extractor.detected_faces():
                filename = faces["filename"]
                detected_faces = faces["detected_faces"]
                if len(detected_faces) > 1:
                    logger.warning("Multiple faces found in %s file: '%s'. Using first detected "
                                   "face.", self.filters[filename]["type"], filename)
                self.filters[filename]["detected_face"] = detected_faces[0]

    def queue_images(self, extractor):
        """ queue images for detection and alignment """
        in_queue = extractor.input_queue
        for fname, img in self.filters.items():
            logger.debug("Adding to filter queue: '%s' (%s)", fname, img["type"])
            feed_dict = dict(filename=fname, image=img["image"])
            if img.get("detected_faces", None):
                feed_dict["detected_faces"] = img["detected_faces"]
            logger.debug("Queueing filename: '%s' items: %s",
                         fname, list(feed_dict.keys()))
            in_queue.put(feed_dict)
        logger.debug("Sending EOF to filter queue")
        in_queue.put("EOF")

    def load_aligned_face(self):
        """ Align the faces for vgg_face input """
        for filename, face in self.filters.items():
            logger.debug("Loading aligned face: '%s'", filename)
            image = face["image"]
            detected_face = face["detected_face"]
            detected_face.load_aligned(image, size=224)
            face["face"] = detected_face.aligned_face
            del face["image"]
            logger.debug("Loaded aligned face: ('%s', shape: %s)",
                         filename, face["face"].shape)

    def get_filter_encodings(self):
        """ Return filter face encodings from Keras VGG Face """
        for filename, face in self.filters.items():
            logger.debug("Getting encodings for: '%s'", filename)
            encodings = self.vgg_face.predict(face["face"])
            logger.debug("Filter Filename: %s, encoding shape: %s", filename, encodings.shape)
            face["encoding"] = encodings
            del face["face"]

    def check(self, detected_face):
        """ Check the extracted Face """
        logger.trace("Checking face with FaceFilter")
        distances = {"filter": list(), "nfilter": list()}
        encodings = self.vgg_face.predict(detected_face.aligned_face)
        for filt in self.filters.values():
            similarity = self.vgg_face.find_cosine_similiarity(filt["encoding"], encodings)
            distances[filt["type"]].append(similarity)

        avgs = {key: avg(val) if val else None for key, val in distances.items()}
        mins = {key: min(val) if val else None for key, val in distances.items()}
        # Filter
        if distances["filter"] and avgs["filter"] > self.threshold:
            msg = "Rejecting filter face: {} > {}".format(round(avgs["filter"], 2), self.threshold)
            retval = False
        # nFilter no Filter
        elif not distances["filter"] and avgs["nfilter"] < self.threshold:
            msg = "Rejecting nFilter face: {} < {}".format(round(avgs["nfilter"], 2),
                                                           self.threshold)
            retval = False
        # Filter with nFilter
        elif distances["filter"] and distances["nfilter"] and mins["filter"] > mins["nfilter"]:
            msg = ("Rejecting face as distance from nfilter sample is smaller: (filter: {}, "
                   "nfilter: {})".format(round(mins["filter"], 2), round(mins["nfilter"], 2)))
            retval = False
        elif distances["filter"] and distances["nfilter"] and avgs["filter"] > avgs["nfilter"]:
            msg = ("Rejecting face as average distance from nfilter sample is smaller: (filter: "
                   "{}, nfilter: {})".format(round(mins["filter"], 2), round(mins["nfilter"], 2)))
            retval = False
        elif distances["filter"] and distances["nfilter"]:
            # k-nn classifier
            var_k = min(5, min(len(distances["filter"]), len(distances["nfilter"])) + 1)
            var_n = sum(list(map(lambda x: x[0],
                                 list(sorted([(1, d) for d in distances["filter"]] +
                                             [(0, d) for d in distances["nfilter"]],
                                             key=lambda x: x[1]))[:var_k])))
            ratio = var_n/var_k
            if ratio < 0.5:
                msg = ("Rejecting face as k-nearest neighbors classification is less than "
                       "0.5: {}".format(round(ratio, 2)))
                retval = False
            else:
                msg = None
                retval = True
        else:
            msg = None
            retval = True
        if msg:
            logger.verbose(msg)
        else:
            logger.trace("Accepted face: (similarity: %s, threshold: %s)",
                         distances, self.threshold)
        return retval
