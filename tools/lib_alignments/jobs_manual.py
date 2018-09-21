#!/usr/bin/env python3
""" Manual processing of alignments """

import platform

import cv2
import numpy as np

from lib.face_alignment import Extract
from . import Annotate, ExtractedFaces, Frames, Rotate


class Interface():
    """ Key controls and interfacing options for OpenCV """
    def __init__(self, alignments, frames):
        self.alignments = alignments
        self.frames = frames
        self.controls = self.set_controls()
        self.state = self.set_state()
        self.skip_mode = {1: "Standard",
                          2: "No Faces",
                          3: "Multi-Faces",
                          4: "Has Faces"}

    def set_controls(self):
        """ Set keyboard controls, destination and help text """
        controls = {"z": {"action": self.iterate_frame,
                          "args": ("navigation", - 1),
                          "help": "Previous Frame"},
                    "x": {"action": self.iterate_frame,
                          "args": ("navigation", 1),
                          "help": "Next Frame"},
                    "[": {"action": self.iterate_frame,
                          "args": ("navigation", - 100),
                          "help": "100 Frames Back"},
                    "]": {"action": self.iterate_frame,
                          "args": ("navigation", 100),
                          "help": "100 Frames Forward"},
                    "{": {"action": self.iterate_frame,
                          "args": ("navigation", "first"),
                          "help": "Go to First Frame"},
                    "}": {"action": self.iterate_frame,
                          "args": ("navigation", "last"),
                          "help": "Go to Last Frame"},
                    27: {"action": "quit",
                         "key_text": "ESC",
                         "args": ("navigation", None),
                         "help": "Exit",
                         "key_type": ord},
                    "/": {"action": self.iterate_state,
                          "args": ("navigation", "frame-size"),
                          "help": "Cycle Frame Zoom"},
                    "s": {"action": self.iterate_state,
                          "args": ("navigation", "skip-mode"),
                          "help": ("Skip Mode (All, No Faces, Multi Faces, "
                                   "Has Faces)")},
                    " ": {"action": self.save_alignments,
                          "key_text": "SPACE",
                          "args": ("edit", None),
                          "help": "Save Alignments"},
                    "r": {"action": self.reload_alignments,
                          "args": ("edit", None),
                          "help": "Reload Alignments (Discard all changes)"},
                    "d": {"action": self.delete_alignment,
                          "args": ("edit", None),
                          "help": "Delete Selected Alignment"},
                    "m": {"action": self.toggle_state,
                          "args": ("edit", "active"),
                          "help": "Change Mode (View, Edit)"},
                    range(10): {"action": self.set_state_value,
                                "key_text": "0 to 9",
                                "args": ["edit", "selected"],
                                "help": "Select/Deselect Face at this Index",
                                "key_type": range},
                    "c": {"action": self.copy_alignments,
                          "args": ("edit", -1),
                          "help": "Copy Previous Frame's Alignments"},
                    "v": {"action": self.copy_alignments,
                          "args": ("edit", 1),
                          "help": "Copy Next Frame's Alignments"},
                    "y": {"action": self.toggle_state,
                          "args": ("image", "display"),
                          "help": "Toggle Image"},
                    "u": {"action": self.iterate_state,
                          "args": ("bounding_box", "color"),
                          "help": "Cycle Bounding Box Color"},
                    "i": {"action": self.iterate_state,
                          "args": ("extract_box", "color"),
                          "help": "Cycle Extract Box Color"},
                    "o": {"action": self.iterate_state,
                          "args": ("landmarks", "color"),
                          "help": "Cycle Landmarks Color"},
                    "p": {"action": self.iterate_state,
                          "args": ("landmarks_mesh", "color"),
                          "help": "Cycle Landmarks Mesh Color"},
                    "h": {"action": self.iterate_state,
                          "args": ("bounding_box", "size"),
                          "help": "Cycle Bounding Box thickness"},
                    "j": {"action": self.iterate_state,
                          "args": ("extract_box", "size"),
                          "help": "Cycle Extract Box thickness"},
                    "k": {"action": self.iterate_state,
                          "args": ("landmarks", "size"),
                          "help": "Cycle Landmarks - point size"},
                    "l": {"action": self.iterate_state,
                          "args": ("landmarks_mesh", "size"),
                          "help": "Cycle Landmarks Mesh - thickness"}}

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
                                "frame-size": 1,
                                "frame_idx": 0,
                                "max_frame": 0,
                                "last_request": 0,
                                "frame_name": None},
                 "edit": {"updated": False,
                          "update_faces": False,
                          "selected": None,
                          "active": 0,
                          "redraw": False}}

        # See lib_alignments/annotate.py for color mapping
        color = 0
        for key in sorted(state.keys()):
            if key not in ("bounding_box", "extract_box", "landmarks",
                           "landmarks_mesh", "image"):
                continue
            state[key]["display"] = True
            if key == "image":
                continue
            color += 1
            state[key]["size"] = 1
            state[key]["color"] = color

        return state

    def save_alignments(self, *args):
        """ Save alignments """
        if not self.state["edit"]["updated"]:
            return
        self.alignments.save_alignments()
        self.state["edit"]["updated"] = False
        self.set_redraw(True)

    def reload_alignments(self, *args):
        """ Reload alignments """
        if not self.state["edit"]["updated"]:
            return
        self.alignments.reload()
        self.state["edit"]["updated"] = False
        self.state["edit"]["update_faces"] = True
        self.set_redraw(True)

    def delete_alignment(self, *args):
        """ Save alignments """
        selected_face = self.get_selected_face_id()
        if self.get_edit_mode() == "View" or selected_face is None:
            return
        frame = self.get_frame_name()
        if self.alignments.delete_alignment_at_index(frame, selected_face):
            self.state["edit"]["selected"] = None
            self.state["edit"]["updated"] = True
            self.state["edit"]["update_faces"] = True
            self.set_redraw(True)

    def copy_alignments(self, *args):
        """ Copy the alignments from the previous or next frame
            to the current frame """
        if self.get_edit_mode() != "Edit":
            return
        frame_id = self.state["navigation"]["frame_idx"] + args[1]
        if not 0 <= frame_id <= self.state["navigation"]["max_frame"]:
            return
        current_frame = self.get_frame_name()
        get_frame = self.frames.file_list_sorted[frame_id]["frame_fullname"]
        alignments = self.alignments.get_alignments_for_frame(get_frame)
        for alignment in alignments:
            self.alignments. add_alignment(current_frame, alignment)
        self.state["edit"]["updated"] = True
        self.state["edit"]["update_faces"] = True
        self.set_redraw(True)

    def toggle_state(self, item, category):
        """ Toggle state of requested item """
        self.state[item][category] = not self.state[item][category]
        self.set_redraw(True)

    def iterate_state(self, item, category):
        """ Cycle through options (6 possible or 3 currently supported) """
        if category == "color":
            max_val = 7
        elif category == "frame-size":
            max_val = 6
        elif category == "skip-mode":
            max_val = 4
        else:
            max_val = 3
        val = self.state[item][category]
        val = val + 1 if val != max_val else 1
        self.state[item][category] = val
        self.set_redraw(True)

    def set_state_value(self, item, category, value):
        """ Set state of requested item or toggle off """
        state = self.state[item][category]
        value = str(value) if value is not None else value
        if state == value:
            self.state[item][category] = None
        else:
            self.state[item][category] = value
        self.set_redraw(True)

    def iterate_frame(self, *args):
        """ Iterate frame up or down, stopping at either end """
        iteration = args[1]
        max_frame = self.state["navigation"]["max_frame"]
        if iteration in ("first", "last"):
            next_frame = 0 if iteration == "first" else max_frame
            self.state["navigation"]["frame_idx"] = next_frame
            self.state["navigation"]["last_request"] = 0
            self.set_redraw(True)
            return

        current_frame = self.state["navigation"]["frame_idx"]
        next_frame = current_frame + iteration
        end = 0 if iteration < 0 else max_frame
        if (max_frame == 0 or
                (end > 0 and next_frame >= end) or
                (end == 0 and next_frame <= end)):
            next_frame = end
        self.state["navigation"]["frame_idx"] = next_frame
        self.state["navigation"]["last_request"] = iteration
        self.set_state_value("edit", "selected", None)

    def get_color(self, item):
        """ Return color for selected item """
        return self.state[item]["color"]

    def get_size(self, item):
        """ Return size for selected item """
        return self.state[item]["size"]

    def get_frame_scaling(self):
        """ Return frame scaling factor for requested item """
        factors = (1, 1.25, 1.5, 2, 0.5, 0.75)
        idx = self.state["navigation"]["frame-size"] - 1
        return factors[idx]

    def get_edit_mode(self):
        """ Return text version and border color for edit mode """
        if self.state["edit"]["active"]:
            return "Edit"
        return "View"

    def get_skip_mode(self):
        """ Return text version of skip mode """
        return self.skip_mode[self.state["navigation"]["skip-mode"]]

    def get_state_color(self):
        """ Return a color based on current state
            white - View Mode
            yellow - Edit Mide
            red - Unsaved alignments """
        color = (255, 255, 255)
        if self.state["edit"]["updated"]:
            color = (0, 0, 255)
        elif self.state["edit"]["active"]:
            color = (0, 255, 255)
        return color

    def get_frame_name(self):
        """ Return the current frame number """
        return self.state["navigation"]["frame_name"]

    def get_selected_face_id(self):
        """ Return the index of the currently selected face """
        try:
            return int(self.state["edit"]["selected"])
        except TypeError:
            return None

    def redraw(self):
        """ Return whether a redraw is required """
        return self.state["edit"]["redraw"]

    def set_redraw(self, request):
        """ Turn redraw requirement on or off """
        self.state["edit"]["redraw"] = request


class Help():
    """ Generate and display help in cli and in window """
    def __init__(self, interface):
        self.interface = interface
        self.helptext = self.generate()

    def generate(self):
        """ Generate help output """
        sections = ("navigation", "display", "color", "size", "edit")
        helpout = {section: list() for section in sections}
        helptext = ""
        for key, val in self.interface.controls.items():
            help_section = val["args"][0]
            if help_section not in ("navigation", "edit"):
                help_section = val["args"][1]
            key_text = val.get("key_text", None)
            key_text = key_text if key_text else key
            helpout[help_section].append((val["help"], key_text))

        helpout["edit"].append(("Bounding Box - Move", "Left Click"))
        helpout["edit"].append(("Bounding Box - Resize", "Middle Click"))

        for section in sections:
            spacer = "=" * int((40 - len(section)) / 2)
            display = "\n{} {} {}\n".format(spacer, section.upper(), spacer)
            helpsection = sorted(helpout[section])
            if section == "navigation":
                helpsection = sorted(helpout[section], reverse=True)
            display += "\n".join("  - '{}': {}".format(item[1], item[0])
                                 for item in helpsection)

            helptext += display
        return helptext

    def render(self):
        """ Render help text to image window """
        image = self.background()
        display_text = self.helptext + self.compile_status()
        self.text_to_image(image, display_text)
        cv2.namedWindow("Help")
        cv2.imshow("Help", image)

    def background(self):
        """ Create an image to hold help text """
        height = 880
        width = 480
        image = np.zeros((height, width, 3), np.uint8)
        color = self.interface.get_state_color()
        cv2.rectangle(image, (0, 0), (width - 1, height - 1),
                      color, 2)
        return image

    def compile_status(self):
        """ Render the status text """
        status = "\n=== STATUS\n"
        navigation = self.interface.state["navigation"]
        frame_scale = int(self.interface.get_frame_scaling() * 100)
        status += "  File: {}\n".format(self.interface.get_frame_name())
        status += "  Frame: {} / {}\n".format(
            navigation["frame_idx"] + 1, navigation["max_frame"] + 1)
        status += "  Frame Size: {}%\n".format(frame_scale)
        status += "  Skip Mode: {}\n".format(self.interface.get_skip_mode())
        status += "  View Mode: {}\n".format(self.interface.get_edit_mode())
        if self.interface.get_selected_face_id() is not None:
            status += "  Selected Face Index: {}\n".format(
                self.interface.get_selected_face_id())
        if self.interface.state["edit"]["updated"]:
            status += "  Warning: There are unsaved changes\n"

        return status

    @staticmethod
    def text_to_image(image, display_text):
        """ Write out and format help text to image """
        pos_y = 0
        for line in display_text.split("\n"):
            if line.startswith("==="):
                pos_y += 10
                line = line.replace("=", "").strip()
            line = line.replace("- '", "[ ").replace("':", " ]")
            cv2.putText(image, line, (20, pos_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
            pos_y += 20


class Manual():
    """ Manually adjust or create landmarks data """
    def __init__(self, alignments, arguments):
        self.arguments = arguments
        self.verbose = arguments.verbose
        self.alignments = alignments
        self.align_eyes = arguments.align_eyes
        self.frames = Frames(arguments.frames_dir, self.verbose)
        self.extracted_faces = None
        self.interface = None
        self.help = None
        self.mouse_handler = None

    def process(self):
        """ Process manual extraction """
        rotate = Rotate(self.alignments, self.arguments,
                        frames=self.frames, child_process=True)
        rotate.process()

        print("\n[MANUAL PROCESSING]")  # Tidy up cli output
        self.extracted_faces = ExtractedFaces(self.frames,
                                              self.alignments,
                                              align_eyes=self.align_eyes)
        self.interface = Interface(self.alignments, self.frames)
        self.help = Help(self.interface)
        self.mouse_handler = MouseHandler(self.interface, self.verbose)

        print(self.help.helptext)
        max_idx = self.frames.count - 1
        self.interface.state["navigation"]["max_frame"] = max_idx
        self.display_frames()

    def display_frames(self):
        """ Iterate through frames """
        cv2.namedWindow("Frame")
        cv2.namedWindow("Faces")
        cv2.setMouseCallback('Frame', self.mouse_handler.on_event)

        frame, faces = self.get_frame()
        press = self.get_keys()

        while True:
            self.help.render()
            cv2.imshow("Frame", frame)
            cv2.imshow("Faces", faces)
            key = cv2.waitKey(1)

            if platform.system() == "Windows" and key == -1:
                break
            elif (platform.system() != "Windows" and
                  cv2.getWindowProperty('Frame', cv2.WND_PROP_VISIBLE) < 1):
                break

            if key in press.keys():
                action = press[key]["action"]
                if action == "quit":
                    break

                if press[key].get("key_type") == range:
                    args = press[key]["args"] + [chr(key)]
                else:
                    args = press[key]["args"]
                action(*args)

            if not self.interface.redraw():
                continue

            frame, faces = self.get_frame()
            self.interface.set_redraw(False)

        cv2.destroyAllWindows()

    def get_keys(self):
        """ Convert keys dict into something useful
            for OpenCV """
        keys = dict()
        for key, val in self.interface.controls.items():
            if val.get("key_type", str) == range:
                for range_key in key:
                    keys[ord(str(range_key))] = val
            elif val.get("key_type", str) == ord:
                keys[key] = val
            else:
                keys[ord(key)] = val

        return keys

    def get_frame(self):
        """ Compile the frame and get faces """
        image = self.frame_selector()
        frame_name = self.interface.get_frame_name()
        alignments = self.alignments.get_alignments_for_frame(frame_name)
        faces_updated = self.interface.state["edit"]["update_faces"]
        roi = self.extracted_faces.get_roi_for_frame(frame_name,
                                                     faces_updated)
        if faces_updated:
            self.interface.state["edit"]["update_faces"] = False

        frame = FrameDisplay(image, alignments, roi, self.interface).image
        faces = self.set_faces(frame_name, alignments).image
        return frame, faces

    def frame_selector(self):
        """ Return frame at given index """
        navigation = self.interface.state["navigation"]
        frame_list = self.frames.file_list_sorted
        frame = frame_list[navigation["frame_idx"]]["frame_fullname"]
        skip_mode = self.interface.get_skip_mode().lower()

        while True:
            if navigation["last_request"] == 0:
                break
            elif navigation["frame_idx"] in (0, navigation["max_frame"]):
                break
            elif skip_mode == "standard":
                break
            elif (skip_mode == "no faces"
                  and not self.alignments.frame_has_faces(frame)):
                break
            elif (skip_mode == "multi-faces"
                  and self.alignments.frame_has_multiple_faces(frame)):
                break
            elif (skip_mode == "has faces"
                  and self.alignments.frame_has_faces(frame)):
                break
            else:
                self.interface.iterate_frame("navigation",
                                             navigation["last_request"])
                frame = frame_list[navigation["frame_idx"]]["frame_fullname"]

        image = self.frames.load_image(frame)
        navigation["last_request"] = 0
        navigation["frame_name"] = frame
        return image

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

            color = self.interface.get_color(item)
            size = self.interface.get_size(item)

            state[item]["display"] = False if color == 7 else True

            if not state[item]["display"]:
                continue

            annotation = getattr(annotate, "draw_{}".format(item))
            annotation(color, size)

        selected_face = self.interface.get_selected_face_id()
        if (selected_face is not None and
                int(selected_face) < len(self.alignments)):
            annotate.draw_grey_out_faces(selected_face)

        self.image = self.resize_frame(annotate.image)

    def resize_frame(self, image):
        """ Set the displayed frame size and add state border"""
        height, width = image.shape[:2]
        color = self.interface.get_state_color()
        cv2.rectangle(image, (0, 0), (width - 1, height - 1),
                      color, 1)

        scaling = self.interface.get_frame_scaling()
        image = cv2.resize(image, (0, 0), fx=scaling, fy=scaling)
        return image


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
        selected_face = self.interface.get_selected_face_id()
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

            if (selected_face is not None
                    and int(selected_face) < len(self.faces)
                    and int(selected_face) != idx):
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
            color = self.interface.get_state_color()
            cv2.rectangle(face, (0, 0), (size - 1, size - 1),
                          color, 1)
            if idx == 0:
                row = face
            else:
                row = np.concatenate((row, face), axis=1)
        return row


class MouseHandler():
    """ Manual Extraction """
    def __init__(self, interface, verbose):
        self.interface = interface
        self.alignments = interface.alignments
        self.frames = interface.frames
        self.extract = Extract(None, "manual",
                               initialize_only=True, verbose=verbose)
        self.mouse_state = None
        self.last_move = None
        self.center = None
        self.dims = None
        self.media = {"frame_id": None,
                      "image": None,
                      "bounding_box": list(),
                      "bounding_last": list(),
                      "bounding_box_orig": list()}

    def on_event(self, event, x, y, flags, param):
        """ Handle the mouse events """
        if self.interface.get_edit_mode() != "Edit":
            return
        elif not self.mouse_state and event not in (cv2.EVENT_LBUTTONDOWN,
                                                    cv2.EVENT_MBUTTONDOWN):
            return

        self.initialize()

        if event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_MBUTTONUP):
            self.mouse_state = None
            self.last_move = None
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_state = "left"
            self.set_bounding_box(x, y)
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.mouse_state = "middle"
            self.set_bounding_box(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mouse_state == "left":
                self.move_bounding_box(x, y)
            elif self.mouse_state == "middle":
                self.resize_bounding_box(x, y)

    def initialize(self):
        """ Update changed parameters """
        frame = self.interface.get_frame_name()
        if frame == self.media["frame_id"]:
            return
        self.media["frame_id"] = frame
        self.media["image"] = self.frames.load_image(frame)
        self.dims = None
        self.center = None
        self.last_move = None
        self.mouse_state = None
        self.media["bounding_box"] = list()
        self.media["bounding_box_orig"] = list()

    def set_bounding_box(self, pt_x, pt_y):
        """ Select or create bounding box """
        if self.interface.get_selected_face_id() is None:
            self.check_click_location(pt_x, pt_y)

        if self.interface.get_selected_face_id() is not None:
            self.dims_from_alignment()
        else:
            self.dims_from_image()

        self.move_bounding_box(pt_x, pt_y)

    def check_click_location(self, pt_x, pt_y):
        """ Check whether the point clicked is within an existing
            bounding box and set face_id """
        frame = self.media["frame_id"]
        alignments = self.alignments.get_alignments_for_frame(frame)

        for idx, alignment in enumerate(alignments):
            left = alignment["x"]
            right = alignment["x"] + alignment["w"]
            top = alignment["y"]
            bottom = alignment["y"] + alignment["h"]

            if left <= pt_x <= right and top <= pt_y <= bottom:
                self.interface.set_state_value("edit", "selected", idx)
                break

    def dims_from_alignment(self):
        """ Set the height and width of bounding box from alignment """
        frame = self.media["frame_id"]
        face_id = self.interface.get_selected_face_id()
        alignment = self.alignments.get_alignments_for_frame(frame)[face_id]
        self.dims = (alignment["w"], alignment["h"])

    def dims_from_image(self):
        """ Set the height and width of bounding
            box at 10% of longest axis """
        size = max(self.media["image"].shape[:2])
        dim = int(size / 10.00)
        self.dims = (dim, dim)

    def bounding_from_center(self):
        """ Get bounding X Y from center """
        pt_x, pt_y = self.center
        width, height = self.dims
        scale = self.interface.get_frame_scaling()
        self.media["bounding_box"] = [int((pt_x / scale) - width / 2),
                                      int((pt_y / scale) - height / 2),
                                      int((pt_x / scale) + width / 2),
                                      int((pt_y / scale) + height / 2)]

    def move_bounding_box(self, pt_x, pt_y):
        """ Move the bounding box """
        self.center = (pt_x, pt_y)
        self.bounding_from_center()
        self.update_landmarks()

    def resize_bounding_box(self, pt_x, pt_y):
        """ Resize the bounding box """
        if not self.last_move:
            self.last_move = (pt_x, pt_y)
            self.media["bounding_box_orig"] = self.media["bounding_box"]

        move_x = int(pt_x - self.last_move[0])
        move_y = int(self.last_move[1] - pt_y)

        original = self.media["bounding_box_orig"]
        updated = self.media["bounding_box"]

        updated[0] = min(self.center[0] - 10, original[0] - move_x)
        updated[1] = min(self.center[1] - 10, original[1] - move_y)
        updated[2] = max(self.center[0] + 10, original[2] + move_x)
        updated[3] = max(self.center[1] + 10, original[3] + move_y)
        self.update_landmarks()
        self.last_move = (pt_x, pt_y)

    def update_landmarks(self):
        """ Update the landmarks """
        self.extract.execute(self.media["image"],
                             manual_face=self.media["bounding_box"])
        landmarks = self.extract.landmarks[0][1]
        left, top, right, bottom = self.media["bounding_box"]
        alignment = {"x": left,
                     "w": right - left,
                     "y": top,
                     "h": bottom - top,
                     "landmarksXY": landmarks}
        frame = self.media["frame_id"]

        if self.interface.get_selected_face_id() is None:
            idx = self.alignments.add_alignment(frame, alignment)
            self.interface.set_state_value("edit", "selected", idx)
        else:
            self.alignments.update_alignment(
                frame,
                self.interface.get_selected_face_id(),
                alignment)
            self.interface.set_redraw(True)

        self.interface.state["edit"]["updated"] = True
        self.interface.state["edit"]["update_faces"] = True
