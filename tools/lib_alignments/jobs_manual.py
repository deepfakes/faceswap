#!/usr/bin/env python3
""" Manual processing of alignments """

import os
import cv2

import numpy as np

from . import Annotate, ExtractedFaces, Faces, Frames


class Interface():
    """ Key controls and interfacing options for OpenCV """
    def __init__(self):
        self.controls = self.set_controls()
        self.state = self.set_state()
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

    @staticmethod
    def set_state():
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

        # See lib_alignments/annotate.py for color mapping
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
        status += "  File: {}\n".format(navigation["frame_name"])
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
        return self.state[item]["color"]

    def get_size(self, item):
        """ Return size for selected item """
        return self.state[item]["size"]


class Manual():
    """ Manually adjust or create landmarks data """
    def __init__(self, alignments, arguments):
        self.verbose = arguments.verbose
        self.alignments = alignments
        self.align_eyes = arguments.align_eyes
        self.faces = self.set_output(arguments.faces_dir)
        self.frames = Frames(arguments.frames_dir, self.verbose)
        self.extracted_faces = ExtractedFaces(self.frames,
                                              self.alignments,
                                              align_eyes=arguments.align_eyes)
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
        frame_list = self.frames.file_list_sorted

        skip_mode = navigation["skip-mode"]
        frame = frame_list[navigation["frame_idx"]]["frame_fullname"]

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
                frame = frame_list[navigation["frame_idx"]]["frame_fullname"]

        image = self.frames.load_image(frame)
        navigation["last_request"] = 0
        navigation["frame_name"] = frame
        return image

    def display_frames(self):
        """ Iterate through frames """
        cv2.namedWindow("Frame")
        cv2.namedWindow("Faces")
        controls = self.interface.controls
        range_keys = dict()

        frame, faces = self.get_frame()
        frame = frame.image
        faces = faces.image

        for keyrange, value in controls.items():
            if not isinstance(keyrange, tuple):
                continue
            for key in range(keyrange[0], keyrange[1] + 1):
                range_keys[key] = value

        while True:
            if cv2.getWindowProperty('Frame', cv2.WND_PROP_VISIBLE) < 1:
                break
            self.interface.render_helptext()
            cv2.imshow("Frames", frame)
            cv2.imshow("Faces", faces)
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

            frame, faces = self.get_frame()
            frame = frame.image
            faces = faces.image

        cv2.destroyAllWindows()

    def get_frame(self):
        """ Compile the frame and get faces """
        image = self.frame_selector()
        state = self.interface.state
        frame_name = state["navigation"]["frame_name"]
        alignments = self.alignments.get_alignments_for_frame(frame_name)
        roi = self.extracted_faces.get_roi_for_frame(frame_name)

        frame_window = FrameDisplay(image, alignments, roi, self.interface)
        faces_window = self.set_faces(frame_name, alignments)

        return frame_window, faces_window

    def set_faces(self, frame, alignments):
        """ Pass the current frame faces to faces window """
        extracted = self.extracted_faces
        size = extracted.size

        faces = extracted.get_faces_for_frame(frame)

        landmarks_xy = [alignment["landmarksXY"] for alignment in alignments]
        landmarks = [
            {"landmarksXY": aligned}
            for aligned
            in extracted.get_aligned_landmarks_for_frame(frame, landmarks_xy)]

        return FacesDisplay(faces, landmarks, size, self.interface)


class FrameDisplay():
    """" Window that holds the frame """
    def __init__(self, image, alignments, roi, interface):
        self.image = image
        self.roi = roi
        self.alignments = alignments
        self.interface = interface
        self.annotate_frame()

    def annotate_frame(self):
        """ Annotate the frame """
        state = self.interface.state
        annotate = Annotate(self.image, self.alignments, self.roi)
        if not state["image"]["display"]:
            annotate.draw_black_image()

        for item in ("bounding_box", "extract_box",
                     "landmarks", "landmarks_mesh"):

            if not state[item]["display"]:
                continue

            color = self.interface.get_color(item)
            size = self.interface.get_size(item)
            annotation = getattr(annotate, "draw_{}".format(item))
            annotation(color, size)

        selected_face = state["navigation"]["select"]
        if (selected_face and
                int(selected_face) < len(self.alignments)):
            annotate.draw_grey_out_faces(selected_face)

        self.image = annotate.image


class FacesDisplay():
    """ Window that holds faces thumbnail """
    def __init__(self, extracted_faces, landmarks, size, interface):
        self.row_length = 4
        self.faces = self.copy_faces(extracted_faces)
        self.roi = self.set_full_roi(size)
        self.landmarks = landmarks
        self.interface = interface

        self.annotate_faces()

        self.image = self.build_faces_image(size)

    @staticmethod
    def copy_faces(faces):
        """ Copy the extracted faces so as not to save the annotations back """
        return [face.copy() for face in faces]

    @staticmethod
    def set_full_roi(size):
        """ ROI is the full frame for faces, so set based on size """
        return [np.array([[(0, 0), (0, size - 1),
                           (size - 1, size - 1), (size - 1, 0)]], np.int32)]

    def annotate_faces(self):
        """ Annotate each of the faces """
        state = self.interface.state
        selected_face = state["navigation"]["select"]
        for idx, face in enumerate(self.faces):
            annotate = Annotate(face, [self.landmarks[idx]], self.roi)
            if not state["image"]["display"]:
                annotate.draw_black_image()

            for item in ("landmarks", "landmarks_mesh"):
                if not state[item]["display"]:
                    continue

                color = self.interface.get_color(item)
                size = self.interface.get_size(item)
                annotation = getattr(annotate, "draw_{}".format(item))
                annotation(color, size)

            if (selected_face and
                    int(selected_face) != idx):
                annotate.draw_grey_out_faces(1)

            self.faces[idx] = annotate.image

    def build_faces_image(self, size):
        """ Display associated faces """
        total_faces = len(self.faces)
        if not total_faces:
            image = self.build_faces_row(list(), size)
            return image
        total_rows = int(total_faces / self.row_length) + 1
        for idx in range(total_rows):
            face_idx = idx * self.row_length
            row_faces = self.faces[face_idx:face_idx + self.row_length]
            if not row_faces:
                break
            row = self.build_faces_row(row_faces, size)
            image = row if idx == 0 else np.concatenate((image, row), axis=0)
        return image

    def build_faces_row(self, faces, size):
        """ Build a row of 4 faces """
        if len(faces) != 4:
            remainder = 4 - (len(faces) % self.row_length)
            for _ in range(remainder):
                faces.append(np.zeros((size, size, 3), np.uint8))
        for idx, face in enumerate(faces):
            cv2.rectangle(face, (0, 0), (size - 1, size - 1),
                          (255, 255, 255), 1)
            if idx == 0:
                row = face
            else:
                row = np.concatenate((row, face), axis=1)
        return row
