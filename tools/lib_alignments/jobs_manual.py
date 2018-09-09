#!/usr/bin/env python3
""" Manual processing of alignments """

import os
import cv2

import numpy as np

from lib.align_eyes import FACIAL_LANDMARKS_IDXS
from plugins.PluginLoader import PluginLoader
from . import DetectedFace, Faces, Frames


class Interface():
    """ Key controls and interfacing options for OpenCV """
    def __init__(self):
        self.controls = self.set_controls()
        self.state = self.set_state()
        self.colors = {1: (255, 0, 0),
                       2: (0, 255, 0),
                       3: (0, 0, 255),
                       4: (255, 255, 0),
                       5: (255, 0, 255),
                       6: (0, 255, 255)}
        self.skip_mode = {1: "Standard",
                          2: "No Faces",
                          3: "Multi-Faces"}
        self.helptext = self.generate_help()

    def set_controls(self):
        """ Set keyboard controls, destination and help text """
        controls = {ord("z"): {"action": self.iterate_frame,
                               "args": ("navigation", - 1),
                               "help": "Previous Frame"},
                    ord("x"): {"action": self.iterate_frame,
                               "args": ("navigation", 1),
                               "help": "Next Frame"},
                    27: {"action": "quit",
                         "args": ("navigation", None),
                         "help": "Exit"},
                    ord("c"): {"action": self.iterate_state,
                               "args": ("navigation", "skip-mode"),
                               "help": ("Set Navigation mode (all, no "
                                        "faces, missing faces)")},
                    ord("y"): {"action": self.toggle_state,
                               "args": ("image", "display"),
                               "help": ("Toggle Image")},
                    ord("u"): {"action": self.toggle_state,
                               "args": ("bounding_box", "display"),
                               "help": ("Toggle Bounding Box")},
                    ord("i"): {"action": self.toggle_state,
                               "args": ("extract_box", "display"),
                               "help": ("Toggle Extract Box")},
                    ord("o"): {"action": self.toggle_state,
                               "args": ("landmarks", "display"),
                               "help": ("Toggle Landmarks")},
                    ord("p"): {"action": self.toggle_state,
                               "args": ("landmarks_mesh", "display"),
                               "help": ("Toggle Landmarks Mesh")},
                    ord("h"): {"action": self.iterate_state,
                               "args": ("bounding_box", "color"),
                               "help": ("Cycle Bounding Box Color")},
                    ord("j"): {"action": self.iterate_state,
                               "args": ("extract_box", "color"),
                               "help": ("Cycle Extract Box Color")},
                    ord("k"): {"action": self.iterate_state,
                               "args": ("landmarks", "color"),
                               "help": ("Cycle Landmarks Color")},
                    ord("l"): {"action": self.iterate_state,
                               "args": ("landmarks_mesh", "color"),
                               "help": ("Cycle Landmarks Mesh Color")},
                    ord("v"): {"action": self.iterate_state,
                               "args": ("bounding_box", "size"),
                               "help": ("Cycle Bounding Box thickness")},
                    ord("b"): {"action": self.iterate_state,
                               "args": ("extract_box", "size"),
                               "help": ("Cycle Extract Box thickness")},
                    ord("n"): {"action": self.iterate_state,
                               "args": ("landmarks", "size"),
                               "help": ("Cycle Landmarks point size")},
                    ord("m"): {"action": self.iterate_state,
                               "args": ("landmarks_mesh", "size"),
                               "help": ("Cycle Landmarks Mesh thickness")},
                    (ord("0"), ord("9")): {
                        "action": self.set_state_value,
                        "args": ["navigation", "select"],
                        "help": "Select/Deselect face at this index"}}

        return controls

    def set_state(self):
        """ Set the initial display state """
        state = {"bounding_box": dict(),
                 "extract_box": dict(),
                 "landmarks": dict(),
                 "landmarks_mesh": dict(),
                 "image": dict(),
                 "navigation": {"skip-mode": 1,
                                "frame_idx": 0,
                                "max_frame": 0,
                                "last_request": 0,
                                "frame_name": None,
                                "select": None}}

        color = 1
        for key in sorted(state.keys()):
            if key == "navigation":
                continue
            state[key]["display"] = bool(key == "image")
            if key == "image":
                continue
            state[key]["size"] = 1
            state[key]["color"] = color
            color = color + 1 if color != 6 else 1

        return state

    def generate_help(self):
        """ Generate help output """
        sections = ("navigation", "display", "color", "size")
        helpout = {section: list() for section in sections}
        helptext = ""
        for key, val in self.controls.items():
            help_section = val["args"][0]
            if help_section != "navigation":
                help_section = val["args"][1]
            if isinstance(key, tuple):
                print(key, val)
                helpout[help_section].append(
                    (val["help"],
                     "{} to {}".format(chr(key[0]), chr(key[1]))))
                continue
            helpout[help_section].append((val["help"], chr(key)))

        for section in sections:
            spacer = "=" * int((40 - len(section)) / 2)
            display = "{} {} {}\n".format(spacer, section.upper(), spacer)
            helpsection = sorted(helpout[section])
            if section == "navigation":
                helpsection = sorted(helpout[section], reverse=True)
            for item in helpsection:
                key = item[1] if item[0] != "Exit" else "ESC"
                display += "  - '{}': {}\n".format(key, item[0])

            helptext += display
        return helptext

    def render_helptext(self):
        """ Render help text to image window """
        image = np.zeros((640, 480, 3), np.uint8)
        pos_y = 10
        status = "=== STATUS\n"
        navigation = self.state["navigation"]
        skip_mode = navigation["skip-mode"]
        status += "  {}\n".format(navigation["frame_name"])
        status += "  Frame: {} / {}\n".format(
            navigation["frame_idx"] + 1, navigation["max_frame"] + 1)
        status += "  Skip-Mode: {}\n".format(self.skip_mode[skip_mode])
        if navigation["select"]:
            status += "  Selected Face Index: {}\n".format(
                navigation["select"])

        display_text = self.helptext + status
        for line in display_text.split("\n"):
            if line.startswith("==="):
                pos_y += 10
                line = line.replace("=", "").strip()
            line = line.replace("- '", "[ ").replace("':", " ]")
            cv2.putText(image, line, (20, pos_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
            pos_y += 20

        cv2.namedWindow("Help")
        cv2.imshow("Help", image)

    def toggle_state(self, item, category):
        """ Toggle state of requested item """
        self.state[item][category] = not self.state[item][category]

    def iterate_state(self, item, category):
        """ Cycle through options (6 possible or 3 currently supported) """
        max_val = 6 if category == "color" else 3
        val = self.state[item][category]
        val = val + 1 if val != max_val else 1
        self.state[item][category] = val

    def set_state_value(self, item, category, value):
        """ Set state of requested item or toggle off """
        state = self.state[item][category]
        if state == value:
            self.state[item][category] = None
        else:
            self.state[item][category] = value

    def iterate_frame(self, *args):
        """ Iterate frame up or down, stopping at either end """
        iteration = args[1]
        current_frame = self.state["navigation"]["frame_idx"]
        end = 0 if iteration < 0 else self.state["navigation"]["max_frame"]
        if current_frame != end:
            self.state["navigation"]["frame_idx"] += iteration
        self.state["navigation"]["last_request"] = iteration

    def get_color(self, item):
        """ Return color for selected item """
        return self.colors[self.state[item]["color"]]

    def get_size(self, item):
        """ Return size for selected item """
        return self.state[item]["size"]


class Manual():
    """ Manually adjust or create landmarks data """
    def __init__(self, alignments, arguments):
        self.verbose = arguments.verbose
        self.alignments = alignments
        self.faces = self.set_output(arguments.faces_dir)
        self.frames = Frames(arguments.frames_dir, self.verbose)
        self.align_eyes = arguments.align_eyes
        self.extractor = PluginLoader.get_extractor("Align")()
        self.interface = Interface()

    def set_output(self, faces_folder):
        """ Set the output to be an existing or new folder """
        if not os.path.isdir(faces_folder):
            print("Creating output folder at {}".format(faces_folder))
            os.makedirs(faces_folder)
        return Faces(faces_folder, self.verbose)

    def process(self):
        """ Process manual extraction """
        print(self.interface.helptext)
        max_idx = self.frames.count - 1
        self.interface.state["navigation"]["max_frame"] = max_idx
        self.display_frames()

    def frame_selector(self):
        """ Return frame at given index """
        navigation = self.interface.state["navigation"]
        alignments = self.alignments
        skip_mode = navigation["skip-mode"]
        frame = self.frames.file_list_sorted[navigation["frame_idx"]]
        while True:
            if skip_mode == 1:
                break
            elif skip_mode == 2 and not alignments.frame_has_faces(frame):
                break
            elif skip_mode == 3 and alignments.frame_has_multiple_faces(frame):
                break
            else:
                iteration = navigation["last_request"]
                old_idx = navigation["frame_idx"]
                self.interface.iterate_frame("navigation", iteration)
                if old_idx == navigation["frame_idx"]:
                    break
                frame = self.frames.file_list_sorted[navigation["frame_idx"]]

        fullpath = os.path.join(self.frames.folder, frame)
        navigation["last_request"] = 0
        navigation["frame_name"] = frame
        return fullpath

    def get_frame(self, rois):
        """ Compile the frame """
        state = self.interface.state
        fullpath = self.frame_selector()
        alignments = self.alignments.alignments.get(
            state["navigation"]["frame_name"], list())
        img = cv2.imread(fullpath)
        if not state["image"]["display"]:
            img = self.black_image(img)
        if state["bounding_box"]["display"]:
            self.draw_bounding_box(img, alignments)
        if state["extract_box"]["display"]:
            self.draw_extract_box(img, rois)
        if state["landmarks"]["display"]:
            self.draw_landmarks(img, alignments)
        if state["landmarks_mesh"]["display"]:
            self.draw_landmarks_mesh(img, alignments)
        if (state["navigation"]["select"] and
                int(state["navigation"]["select"]) < len(alignments)):
            self.grey_out_faces(img, rois)
        return img, alignments

    @staticmethod
    def black_image(image):
        """ Return black image to correct dimensions """
        height, width = image.shape[:2]
        image = np.zeros((height, width, 3), np.uint8)
        return image

    def draw_bounding_box(self, image, alignments):
        """ Draw the bounding box around faces """
        color = self.interface.get_color("bounding_box")
        thickness = self.interface.get_size("bounding_box")
        for alignment in alignments:
            top_left = (alignment["x"], alignment["y"])
            bottom_right = (alignment["x"] + alignment["w"],
                            alignment["y"] + alignment["h"])
            cv2.rectangle(image, top_left, bottom_right, color, thickness)

    def draw_extract_box(self, image, rois):
        """ Draw the extracted face box """
        if not rois:
            return
        color = self.interface.get_color("extract_box")
        thickness = self.interface.get_size("extract_box")

        for idx, roi in enumerate(rois):
            top_left = [point for point in roi[0].squeeze()[0]]
            top_left = (top_left[0], top_left[1] - 10)

            cv2.putText(image, str(idx), top_left,
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, color, thickness)
            cv2.polylines(image, roi, True, color, thickness)

    def draw_landmarks(self, image, alignments):
        """ Draw the facial landmarks """
        color = self.interface.get_color("landmarks")
        radius = self.interface.get_size("landmarks")

        for alignment in alignments:
            landmarks = alignment["landmarksXY"]
            for (pos_x, pos_y) in landmarks:
                cv2.circle(image, (pos_x, pos_y), radius, color, -1)

    def draw_landmarks_mesh(self, image, alignments):
        """ Draw the facial landmarks """
        color = self.interface.get_color("landmarks_mesh")
        thickness = self.interface.get_size("landmarks_mesh")

        for alignment in alignments:
            landmarks = alignment["landmarksXY"]
            for key, val in FACIAL_LANDMARKS_IDXS.items():
                points = np.array([landmarks[val[0]:val[1]]], np.int32)
                fill_poly = bool(key in ("right_eye", "left_eye", "mouth"))
                cv2.polylines(image, points, fill_poly, color, thickness)

    def grey_out_faces(self, image, rois):
        """ Grey out all faces except target """
        overlay = image.copy()
        for idx, roi in enumerate(rois):
            if idx != int(self.interface.state["navigation"]["select"]):
                cv2.fillPoly(overlay, roi, (0, 0, 0))
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    def display_frames(self):
        """ Iterate through frames """
        cv2.namedWindow("Frame")

        controls = self.interface.controls
        matrices = list()
        range_keys = dict()

        img, alignments = self.get_frame(matrices)

        for keyrange, value in controls.items():
            if not isinstance(keyrange, tuple):
                continue
            for key in range(keyrange[0], keyrange[1] + 1):
                range_keys[key] = value

        while True:
            if cv2.getWindowProperty('Frame', cv2.WND_PROP_VISIBLE) < 1:
                break
            self.interface.render_helptext()
            rois = self.get_faces(img, alignments)
            cv2.imshow("Frames", img)
            key = cv2.waitKey(100)

            if key in controls.keys():
                action = controls[key]["action"]
                args = controls[key]["args"]
                if action == "quit":
                    break
                else:
                    action(*args)
            elif key in range_keys.keys():
                action = range_keys[key]["action"]
                args = range_keys[key]["args"] + [chr(key)]
                action(*args)

            img, alignments = self.get_frame(rois)

        cv2.destroyAllWindows()

    def get_faces(self, image, frame_alignments):
        """ Display associated face """
        size = 256  # Standard Faceswap size
        faces = list()
        rois = list()
        total_alignments = len(frame_alignments)

        for idx, alignment in enumerate(frame_alignments):
            face = DetectedFace(image,
                                alignment["r"],
                                alignment["x"],
                                alignment["w"],
                                alignment["y"],
                                alignment["h"],
                                alignment["landmarksXY"])

            resized_face, f_align = self.extractor.extract(image,
                                                           face,
                                                           size,
                                                           self.align_eyes)
            rois.append(self.original_roi(f_align, size))

            if idx % 4 == 0:
                row = resized_face
            else:
                row = np.concatenate((row, resized_face), axis=1)

            if (idx + 1) % 4 == 0 or idx + 1 == total_alignments:
                faces.append(row)

        image = self.compile_faces_image(faces, total_alignments, size)
        self.show_hide_faces(bool(faces), image)
        return rois

    @staticmethod
    def original_roi(matrix, size):
        """ Return the original ROI of an extracted face """
        padding = 48    # Faceswap padding

        points = np.array([[0, 0], [0, size - 1],
                           [size - 1, size - 1], [size - 1, 0]],
                          np.int32)
        points = points.reshape((-1, 1, 2))

        matrix = matrix * (size - 2 * padding)
        matrix[:, 2] += padding
        matrix = cv2.invertAffineTransform(matrix)

        return [cv2.transform(points, matrix)]

    @staticmethod
    def compile_faces_image(faces, total_faces, size):
        """ Compile the faces into tiled image """
        image = None
        blank_face = np.zeros((size, size, 3), np.uint8)
        total_rows = len(faces)
        remainder = 4 - (total_faces % 4)
        for idx, row in enumerate(faces):
            if idx + 1 == total_rows and remainder != 4:
                for _ in range(remainder):
                    row = np.concatenate((row, blank_face), axis=1)
            if idx == 0:
                image = row
                continue
            image = np.concatenate((image, row), axis=0)
        return image

    @staticmethod
    def show_hide_faces(display, image):
        """ Show or remove window if faces are available """
        if display:
            cv2.namedWindow("Faces")
            cv2.imshow("Faces", image)
        elif cv2.getWindowProperty("Faces", cv2.WND_PROP_VISIBLE) == 1.0:
            cv2.destroyWindow("Faces")
