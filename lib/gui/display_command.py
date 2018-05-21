#!/usr/bin python3
""" Command specific tabs of Display Frame of the Faceswap GUI """
import datetime
import os
import tkinter as tk

from tkinter import ttk


from .display_page import DisplayOptionalPage
from .tooltip import Tooltip
from .utils import Images, FileHandler
from .display_graph import TrainingGraph

class PreviewExtract(DisplayOptionalPage):
    """ Tab to display output preview images for extract and convert """

    def display_item_set(self):
        """ Load the latest preview if available """
        Images().load_latest_preview()
        self.display_item = Images().previewoutput

    def display_item_process(self):
        """ Display the preview """
        if not self.subnotebook.children:
            self.add_child()
        else:
            self.update_child()

    def add_child(self):
        """ Add the preview label child """
        preview = self.subnotebook_add_page(self.tabname, widget=None)
        lblpreview = ttk.Label(preview, image=Images().previewoutput[1])
        lblpreview.pack(side=tk.TOP, anchor=tk.NW)
        Tooltip(lblpreview, text=self.helptext, wraplength=200)

    def update_child(self):
        """ Update the preview image on the label """
        for widget in self.subnotebook_get_widgets():
            widget.configure(image=Images().previewoutput[1])

    def save_items(self):
        """ Open save dialogue and save preview """
        location = FileHandler('dir').retfile
        if not location:
            return
        filename = 'extract_convert_preview'
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(location, '{}_{}.{}'.format(filename, now, 'png'))
        Images().previewoutput[0].save(filename)
        print('Saved preview to {}'.format(filename))

class PreviewTrain(DisplayOptionalPage):
    """ Training preview image(s) """

    def display_item_set(self):
        """ Load the latest preview if available """
        Images().load_training_preview()
        self.display_item = Images().previewtrain

    def display_item_process(self):
        """ Display the preview(s) resized as appropriate """
        sortednames = sorted([name for name in Images().previewtrain.keys()])
        existing = self.subnotebook_get_titles_ids()

        for name in sortednames:
            if name not in existing.keys():
                self.add_child(name)
            else:
                tab_id = existing[name]
                self.update_child(tab_id, name)

    def add_child(self, name):
        """ Add the preview canvas child """
        preview = PreviewTrainCanvas(self.subnotebook, name)
        preview = self.subnotebook_add_page(name, widget=preview)
        Tooltip(preview, text=self.helptext, wraplength=200)
        self.vars['modified'].set(Images().previewtrain[name][2])

    def update_child(self, tab_id, name):
        """ Update the preview canvas """
        if self.vars['modified'].get() != Images().previewtrain[name][2]:
            self.vars['modified'].set(Images().previewtrain[name][2])
            widget = self.subnotebook_page_from_id(tab_id)
            widget.reload()

    def save_items(self):
        """ Open save dialogue and save preview """
        location = FileHandler('dir').retfile
        if not location:
            return
        for preview in self.subnotebook.children.values():
            preview.save_preview(location)

class PreviewTrainCanvas(ttk.Frame):
    """ Canvas to hold a training preview image """
    def __init__(self, parent, previewname):
        ttk.Frame.__init__(self, parent)

        self.name = previewname
        Images().resize_image(self.name, None)
        self.previewimage = Images().previewtrain[self.name][1]

        self.canvas = tk.Canvas(self, bd=0, highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.imgcanvas = self.canvas.create_image(0, 0, image=self.previewimage, anchor=tk.NW)
        self.bind("<Configure>", self.resize)

    def resize(self, event):
        """  Resize the image to fit the frame, maintaining aspect ratio """
        framesize = (event.width, event.height)
        # Sometimes it tries to resize the image before the frame has been drawn
        framesize = None if framesize == (1, 1) else framesize
        Images().resize_image(self.name, framesize)
        self.reload()

    def reload(self):
        """ Reload the preview image """
        self.previewimage = Images().previewtrain[self.name][1]
        self.canvas.itemconfig(self.imgcanvas, image=self.previewimage)

    def save_preview(self, location):
        """ Save the figure to file """
        filename = self.name
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(location, '{}_{}.{}'.format(filename, now, 'png'))
        Images().previewtrain[self.name][0].save(filename)
        print('Saved preview to {}'.format(filename))

class GraphDisplay(DisplayOptionalPage):
    """ The Graph Tab of the Display section """

    def display_item_set(self):
        """ Load the graph(s) if available """
        self.display_item = self.session.lossdict

    def display_item_process(self):
        """ Add a single graph to the graph window """
        losskeys = sorted([key for key in self.session.lossdict.keys()])
        tabcount = int(len(self.session.lossdict) / 2)
        existing = self.subnotebook_get_titles_ids()

        for i in range(tabcount):
            selectedkeys = losskeys[i * 2:(i + 1) * 2]
            name = ' - '.join(selectedkeys).title().replace('_', ' ')
            if name not in existing.keys():
                selectedloss = {key: self.session.lossdict[key] for key in selectedkeys}
                self.add_child(name, selectedkeys, selectedloss)

    def add_child(self, name, keys, loss):
        """ Add the graph for the selected keys """
        graph = TrainingGraph(self.subnotebook, loss, keys)
        graph.build()
        graph = self.subnotebook_add_page(name, widget=graph)
        Tooltip(graph, text=self.helptext, wraplength=200)

    def save_items(self):
        """ Open save dialogue and save graphs """
        graphlocation = FileHandler('dir').retfile
        if not graphlocation:
            return
        for graph in self.subnotebook.children.values():
            graph.save_fig(graphlocation)
