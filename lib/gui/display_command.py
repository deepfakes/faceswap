#!/usr/bin python3
""" Command specific tabs of Display Frame of the Faceswap GUI """
import datetime
import os
import tkinter as tk

from tkinter import ttk
from math import ceil, floor

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt, style
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy

from .display_page import DisplayOptionalPage
from .tooltip import Tooltip
from .utils import Images, FileHandler

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
        graph = Graph(self.subnotebook, loss, keys)
        graph = self.subnotebook_add_page(name, widget=graph)
        Tooltip(graph, text=self.helptext, wraplength=200)

    def save_items(self):
        """ Open save dialogue and save graphs """
        graphlocation = FileHandler('dir').retfile
        if not graphlocation:
            return
        for graph in self.subnotebook.children.values():
            graph.save_fig(graphlocation)

class Graph(ttk.Frame):
    """ Each graph to be displayed. Until training is run it is not known
        how many graphs will be required, so they sit in their own class
        ready to be created when requested """

    def __init__(self, parent, loss, losskeys):
        ttk.Frame.__init__(self, parent)

        self.loss = (losskeys, loss)
        self.anim = None
        self.plotcanvas = None

        self.ylim = (100, 0)

        style.use('ggplot')

        self.fig = plt.figure(figsize=(4, 4), dpi=75)
        self.ax1 = self.fig.add_subplot(1, 1, 1)
        self.lines = {'losslines': list(), 'trndlines': list()}
        self.build_graph()

    def build_graph(self):
        """ Update the plot area with loss values and cycle through to
        animate """
        self.ax1.set_xlabel('Iterations')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_ylim(0.00, 0.01)
        self.ax1.set_xlim(0, 1)

        losslbls = [lbl.replace('_', ' ').title() for lbl in self.loss[0]]
        for idx, linecol in enumerate(['blue', 'red']):
            self.lines['losslines'].extend(self.ax1.plot(0, 0,
                                                         color=linecol,
                                                         linewidth=1,
                                                         label=losslbls[idx]))
        for idx, linecol in enumerate(['navy', 'firebrick']):
            lbl = losslbls[idx]
            lbl = 'Trend{}'.format(lbl[lbl.rfind(' '):])
            self.lines['trndlines'].extend(self.ax1.plot(0, 0,
                                                         color=linecol,
                                                         linewidth=2,
                                                         label=lbl))

        self.ax1.legend(loc='upper right')

        plt.subplots_adjust(left=0.075, bottom=0.075, right=0.95, top=0.95,
                            wspace=0.2, hspace=0.2)

        self.plotcanvas = FigureCanvasTkAgg(self.fig, self)
        self.plotcanvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.anim = animation.FuncAnimation(self.fig, self.animate, interval=200, blit=False)
        self.plotcanvas.draw()

    def animate(self, i):
        """ Read loss data and apply to graph """
        loss = [self.loss[1][key][:] for key in self.loss[0]]

        xlim = self.recalculate_axes(loss)

        self.set_animation_rate(xlim)

        xrng = [x for x in range(xlim)]

        self.raw_plot(xrng, loss)

        if xlim > 10:
            self.trend_plot(xrng, loss)

    def recalculate_axes(self, loss):
        ''' Recalculate the latest x and y axes limits from latest data '''
        ymin = floor(min([min(lossvals) for lossvals in loss]) * 1000) / 1000
        ymax = ceil(max([max(lossvals) for lossvals in loss]) * 1000) / 1000

        if ymin < self.ylim[0] or ymax > self.ylim[1]:
            self.ylim = (ymin, ymax)
            self.ax1.set_ylim(self.ylim[0], self.ylim[1])

        xlim = len(loss[0])
        xlim = 2 if xlim == 1 else xlim
        self.ax1.set_xlim(0, xlim - 1)

        return xlim

    def set_animation_rate(self, iterations):
        """ Change the animation update interval based on how
            many iterations have been
            There's no point calculating a graph over thousands of
            points of data when the change will be miniscule """
        if iterations > 30000:
            speed = 60000           #1 min updates
        elif iterations > 20000:
            speed = 30000           #30 sec updates
        elif iterations > 10000:
            speed = 10000           #10 sec updates
        elif iterations > 5000:
            speed = 5000            #5 sec updates
        elif iterations > 1000:
            speed = 2000            #2 sec updates
        elif iterations > 500:
            speed = 1000            #1 sec updates
        elif iterations > 100:
            speed = 500             #0.5 sec updates
        else:
            speed = 200             #200ms updates
        if not self.anim.event_source.interval == speed:
            self.anim.event_source.interval = speed

    def raw_plot(self, x_range, loss):
        ''' Raw value plotting '''
        for idx, lossvals in enumerate(loss):
            self.lines['losslines'][idx].set_data(x_range, lossvals)

    def trend_plot(self, x_range, loss):
        ''' Trend value plotting '''
        for idx, lossvals in enumerate(loss):
            fit = numpy.polyfit(x_range, lossvals, 3)
            poly = numpy.poly1d(fit)
            self.lines['trndlines'][idx].set_data(x_range, poly(x_range))

    def save_fig(self, location):
        """ Save the figure to file """
        filename = ' - '.join(self.loss[0])
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(location, '{}_{}.{}'.format(filename, now, 'png'))
        self.fig.set_size_inches(16, 9)
        self.fig.savefig(filename, bbox_inches='tight', dpi=120)
        print('Saved graph to {}'.format(filename))
        self.resize_fig()

    def resize_fig(self):
        """ Resize the figure back to the canvas """
        class Event(object):
            """ Event class that needs to be passed to
                plotcanvas.resize """
            pass
        Event.width = self.winfo_width()
        Event.height = self.winfo_height()
        self.plotcanvas.resize(Event)
