#!/usr/bin/env python3
""" Manual processing of alignments """

import cv2

import numpy as np

from . import Annotate, ExtractedFaces, Frames


class Interface():
    """ Key controls and interfacing options for OpenCV """
    def __init__(self, alignments):
        self.alignments = alignments
        self.controls = self.set_controls()
        self.state = self.set_state()
        self.skip_mode = {1: "Standard",
                          2: "No Faces",
                          3: "Multi-Faces",
                          4: "Has Faces"}

    def set_controls(self):
        """ Set keyboard controls, destination and help text """
        controls = {ord("["): {"action": self.iterate_frame,
                               "args": ("navigation", - 1),
                               "help": "Previous Frame"},
                    ord("]"): {"action": self.iterate_frame,
                               "args": ("navigation", 1),
                               "help": "Next Frame"},
                    ord("{"): {"action": self.iterate_frame,
                               "args": ("navigation", - 100),
                               "help": "100 Frames Back"},
                    ord("}"): {"action": self.iterate_frame,
                               "args": ("navigation", 100),
                               "help": "100 Frames Forward"},
                    ord("<"): {"action": self.iterate_frame,
                               "args": ("navigation", "first"),
                               "help": "Go to First Frame"},
                    ord(">"): {"action": self.iterate_frame,
                               "args": ("navigation", "last"),
                               "help": "Go to Last Frame"},
                    27: {"action": "quit",
                         "key_text": "ESC",
                         "args": ("navigation", None),
                         "help": "Exit"},
                    ord("#"): {"action": self.iterate_state,
                               "args": ("navigation", "frame-size"),
                               "help": "Cycle Frame Size"},
                    ord("c"): {"action": self.iterate_state,
                               "args": ("navigation", "skip-mode"),
                               "help": ("Skip Mode (All, No Faces, Multi "
                                        "Faces, Has Faces)")},
                    32: {"action": self.save_alignments,
                         "key_text": "SPACE",
                         "args": ("edit", None),
                         "help": "Save Alignments"},
                    ord("r"): {"action": self.reload_alignments,
                               "args": ("edit", None),
                               "help": "Reload Alignments (Discard changes)"},
                    ord("d"): {"action": self.delete_alignment,
                               "args": ("edit", None),
                               "help": "Delete Selected Alignment"},
                    ord("m"): {"action": self.toggle_state,
                               "args": ("edit", "active"),
                               "help": "Change Mode (View, Edit)"},
                    (ord("0"), ord("9")): {
                        "action": self.set_state_value,
                        "key_text": "0 to 9",
                        "args": ["edit", "selected"],
                        "help": "Select/Deselect Face at this Index"},
                    ord("y"): {"action": self.toggle_state,
                               "args": ("image", "display"),
                               "help": "Toggle Image"},
                    ord("u"): {"action": self.iterate_state,
                               "args": ("bounding_box", "color"),
                               "help": "Cycle Bounding Box Color"},
                    ord("i"): {"action": self.iterate_state,
                               "args": ("extract_box", "color"),
                               "help": "Cycle Extract Box Color"},
                    ord("o"): {"action": self.iterate_state,
                               "args": ("landmarks", "color"),
                               "help": "Cycle Landmarks Color"},
                    ord("p"): {"action": self.iterate_state,
                               "args": ("landmarks_mesh", "color"),
                               "help": "Cycle Landmarks Mesh Color"},
                    ord("h"): {"action": self.iterate_state,
                               "args": ("bounding_box", "size"),
                               "help": "Cycle Bounding Box thickness"},
                    ord("j"): {"action": self.iterate_state,
                               "args": ("extract_box", "size"),
                               "help": "Cycle Extract Box thickness"},
                    ord("k"): {"action": self.iterate_state,
                               "args": ("landmarks", "size"),
                               "help": "Cycle Landmarks - point size"},
                    ord("l"): {"action": self.iterate_state,
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
                          "redraw": False},
                 "frame": {"image": None},
                 "faces": {"image": None}}

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
        edit_mode = self.get_edit_mode().lower()
        selected_face = self.get_selected_face_id()
        if edit_mode == "view" or not selected_face:
            return
        frame = self.get_frame_name()
        if self.alignments.delete_alignment_at_index(frame, selected_face):
            self.state["edit"]["selected"] = None
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
            return

        current_frame = self.state["navigation"]["frame_idx"]
        next_frame = current_frame + iteration
        end = 0 if iteration < 0 else max_frame
        if (end > 0 and next_frame >= end) or (end == 0 and next_frame <= end):
            next_frame = end
        self.state["navigation"]["frame_idx"] = next_frame
        self.state["navigation"]["last_request"] = iteration
        self.set_redraw(True)

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

    def get_frame_image(self):
        """ Return the current frame number """
        return self.state["frame"]["image"]

    def set_frame_image(self, image):
        """ Return the current frame number """
        self.state["frame"]["image"] = image

    def get_selected_face_id(self):
        """ Return the index of the currently selected face """
        return self.state["edit"]["selected"]

    def get_faces_image(self):
        """ Return the current frame number """
        return self.state["faces"]["image"]

    def set_faces_image(self, image):
        """ Return the current frame number """
        self.state["faces"]["image"] = image

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
            key_text = key_text if key_text else chr(key)
            helpout[help_section].append((val["help"], key_text))

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
        height = 800
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
        status += "  Skip-Mode: {}\n".format(self.interface.get_skip_mode())
        status += "  View-Mode: {}\n".format(self.interface.get_edit_mode())
        if self.interface.get_selected_face_id():
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
        self.verbose = arguments.verbose
        self.alignments = alignments
        self.align_eyes = arguments.align_eyes
        self.frames = Frames(arguments.frames_dir, self.verbose)
        self.extracted_faces = ExtractedFaces(self.frames,
                                              self.alignments,
                                              align_eyes=arguments.align_eyes)
        self.interface = Interface(self.alignments)
        self.help = Help(self.interface)
#        self.mouse_handler = MouseHandler(self.interface, self.alignments)

    def process(self):
        """ Process manual extraction """
        print(self.help.helptext)
        max_idx = self.frames.count - 1
        self.interface.state["navigation"]["max_frame"] = max_idx
        self.display_frames()

    def display_frames(self):
        """ Iterate through frames """
        cv2.namedWindow("Frame")
        cv2.namedWindow("Faces")
#        cv2.setMouseCallback('Frame', self.mouse_handler.on_event)

        controls = self.interface.controls
        range_keys = dict()

        self.get_frame()

        for keyrange, value in controls.items():
            if not isinstance(keyrange, tuple):
                continue
            for key in range(keyrange[0], keyrange[1] + 1):
                range_keys[key] = value

        while True:
            if cv2.getWindowProperty('Frame', cv2.WND_PROP_VISIBLE) < 1:
                break
            self.help.render()
            cv2.imshow("Frame", self.interface.get_frame_image())
            cv2.imshow("Faces", self.interface.get_faces_image())
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
            if not self.interface.redraw():
                continue

            self.get_frame()
            self.interface.set_redraw(False)

        cv2.destroyAllWindows()

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

        self.interface.set_frame_image(FrameDisplay(image,
                                                    alignments,
                                                    roi,
                                                    self.interface).image)
        self.interface.set_faces_image(self.set_faces(frame_name,
                                                      alignments).image)

    def frame_selector(self):
        """ Return frame at given index """
        navigation = self.interface.state["navigation"]
        frame_list = self.frames.file_list_sorted
        frame = frame_list[navigation["frame_idx"]]["frame_fullname"]
        skip_mode = self.interface.get_skip_mode().lower()

        while True:
            if skip_mode == "standard":
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
        if (selected_face and
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

            if (selected_face
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
    def __init__(self, interface):
        self.interface = interface
        self.left_down = False
        self.rect_drawn = False

    def on_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_down = True
        if event == cv2.EVENT_LBUTTONUP:
            self.left_down = False
            print("LB_UP")
            print("x", x)
            print("y", y)
            print("flags", flags)
            print("param", param)
        if event == cv2.EVENT_MOUSEWHEEL:
            print("MouseWheel")
            print("x", x)
            print("y", y)
            print("flags", flags)
            print("param", param)
        if event == cv2.EVENT_MOUSEMOVE and self.left_down:
            self.action_lbutton(x, y)

    def action_lbutton(self, pt_x, pt_y):
        """ Create or select existing bounding box """
        if not self.left_down:
            return
        image = self.interface.get_frame_image()
        if not self.rect_drawn:
            top_left = (pt_x - 128, pt_y - 128)
            bottom_right = (pt_x + 128, pt_y + 128)
            print(top_left)
            print(bottom_right)
            print(self.rect_drawn)
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 3)
            self.rect_drawn = True
            print(self.rect_drawn)
