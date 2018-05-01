#!/usr/bin python3
""" Display Frame of the Faceswap GUI

    What is displayed in the Display Frame varies
    depending on what tasked is being run """
import os
import tkinter as tk
from tkinter import TclError, ttk
from math import ceil, floor

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt, style
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy
from PIL import Image, ImageTk

from .utils import Images
from .wrapper import ProcessWrapper

class DisplayNotebook(ttk.Notebook):
    """ The display tabs """

    def __init__(self, parent):
        ttk.Notebook.__init__(self, parent, width=780)
        parent.add(self)

        self.images = Images()
        self.add_static_tabs()
        self.static_tabs = [child for child in self.tabs()]
        print(self.static_tabs)

    def add_static_tabs(self):
        """ Add tabs that are permanently available """
        for tab in ('job queue', 'analysis'):
            frame = self.add_frame()

            self.add(frame, text=tab.title())

    def add_frame(self):
        """ Add a single frame for holding tab's contents """
        frame = ttk.Frame(self)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        return frame

    def command_display(self, command):
        """ Select what to display based on incoming
            command """
        build_tabs = getattr(self, '{}_tabs'.format(command))()
        build_tabs()

    def extract_tabs(self):
        """ Build the extract tabs """
        for tab in "preview":
            frame = self.add_frame()

            if tab == 'preview':
                preview = PreviewDisplay(frame, self.images.pathpreview)
                preview.update_preview()

            self.add(frame, text=tab.title())

    def train_tabs(self):
        """ Build the train tabs """
        for tab in ("graph", "preview"):
            frame = self.add_frame()

            if tab == 'graph':
                graphframe = GraphDisplay(frame)
                graphframe.create_graphs()
            elif tab == 'preview':
                preview = PreviewDisplay(frame, self.images.pathpreview)
                preview.update_preview()

            self.add(frame, text=tab.title())

    def convert_tabs(self):
        """ Build the convert tabs
            Currently identical to Extract, so just call that """
        self.extract_tabs()

    def remove_tabs(self):
        """ Remove all command specific tabs """
        for child in self.tabs():
            if child not in self.static_tabs:
                self.forget(child)

class GraphDisplay(object):
    """ The Graph Tab of the Display section """

    def __init__(self, frame):
        self.frame = frame
        self.wrapper = ProcessWrapper()
        self.losskeys = None

        self.graphpane = tk.PanedWindow(self.frame, sashrelief=tk.RAISED, orient=tk.VERTICAL)
        self.graphpane.pack(fill=tk.BOTH, expand=True)

        self.graphs = list()

    def create_graphs(self):
        """ create the graph frames when there are loss values to graph """
        if not self.wrapper.lossdict:
            self.frame.after(1000, self.create_graphs)
            return

        self.losskeys = sorted([key for key in self.wrapper.lossdict.keys()])

        framecount = int(len(self.wrapper.lossdict) / 2)
        for i in range(framecount):
            self.add_graph(i)

        self.monitor_state()

    def add_graph(self, index):
        """ Add a single graph to the graph window """
        graphframe = ttk.Frame(self.graphpane)
        self.graphpane.add(graphframe)

        selectedkeys = self.losskeys[index * 2:(index + 1) * 2]
        selectedloss = {key: self.wrapper.lossdict[key] for key in selectedkeys}

        graph = Graph(graphframe, selectedloss, selectedkeys)
        self.graphs.append(graph)
        graph.build_graph()

    def monitor_state(self):
        """ Check there is a task still running. If not, destroy graphs
            and reset graph display to waiting state """
        if self.wrapper.lossdict:
            self.frame.after(5000, self.monitor_state)
            return
        self.destroy_graphs()
        self.create_graphs()

    def destroy_graphs(self):
        """ Destroy graphs when the process has stopped """
        for graph in self.graphs:
            del graph
        self.graphs = list()
        for child in self.graphpane.panes():
            self.graphpane.remove(child)

class Graph(object):
    """ Each graph to be displayed. Until training is run it is not known
        how many graphs will be required, so they sit in their own class
        ready to be created when requested """

    def __init__(self, frame, loss, losskeys):
        self.frame = frame
        self.loss = loss
        self.losskeys = losskeys
        self.anim = None

        self.ylim = (100, 0)

        style.use('ggplot')

        self.fig = plt.figure(figsize=(4, 4), dpi=75)
        self.ax1 = self.fig.add_subplot(1, 1, 1)
        self.lines = {'losslines': list(), 'trndlines': list()}

    def build_graph(self):
        """ Update the plot area with loss values and cycle through to
        animate """
        self.ax1.set_xlabel('Iterations')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_ylim(0.00, 0.01)
        self.ax1.set_xlim(0, 1)

        losslbls = [lbl.replace('_', ' ').title() for lbl in self.losskeys]
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

        plotcanvas = FigureCanvasTkAgg(self.fig, self.frame)
        plotcanvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.anim = animation.FuncAnimation(self.fig, self.animate, interval=200, blit=False)
        plotcanvas.draw()

    def animate(self, i):
        """ Read loss data and apply to graph """
        loss = [self.loss[key][:] for key in self.losskeys]

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

class PreviewDisplay(object):
    """ The Preview tab of the Display section """

    def __init__(self, frame, pathpreview):
        self.canvas = tk.Canvas(frame, bd=0, highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.framesize = None
        self.imgoriginal = None
        self.previewimg = None
        self.errcount = 0
        self.pathpreview = os.path.join(pathpreview, '.gui_preview.png')

        self.imgcanvas = self.canvas.create_image(0, 0, image=self.previewimg, anchor=tk.NW)
        frame.bind("<Configure>", self.resize)

    def update_preview(self):
        """ Display the image if it exists or a place holder if it doesn't """
        self.load_preview()
        self.canvas.itemconfig(self.imgcanvas, image=self.previewimg)
        self.canvas.after(5000, self.update_preview)

    def load_preview(self):
        """ Load the preview image into tk PhotoImage """
        if os.path.exists(self.pathpreview):
            try:
                self.imgoriginal = Image.open(self.pathpreview)
                self.display_image()
                self.errcount = 0
            except (ValueError, TclError):
                # This is probably an error reading the file whilst it's
                # being saved  so ignore it for now and only pick up if
                # there have been multiple consecutive fails
                if self.errcount < 10:
                    self.errcount += 1
                    self.previewimg = None
                else:
                    print('Error reading the preview file')
        else:
            self.imgoriginal = None
            self.previewimg = None

    def display_image(self):
        """ Set the display image size based on the current frame size """
        if not self.framesize:
            displayimg = self.imgoriginal
        else:
            frameratio = float(self.framesize[0]) / float(self.framesize[1])
            imgratio = float(self.imgoriginal.size[0]) / float(self.imgoriginal.size[1])

            if frameratio <= imgratio:
                scale = self.framesize[0] / float(self.imgoriginal.size[0])
                size = (self.framesize[0], int(self.imgoriginal.size[1] * scale))
            else:
                scale = self.framesize[1] / float(self.imgoriginal.size[1])
                size = (int(self.imgoriginal.size[0] * scale), self.framesize[1])

            displayimg = self.imgoriginal.resize(size, Image.ANTIALIAS)

        self.previewimg = ImageTk.PhotoImage(displayimg)

    def resize(self, event):
        """  Resize the image to fit the frame, maintaining aspect ratio """
        self.framesize = (event.width, event.height)
        if self.imgoriginal:
            self.display_image()
        else:
            self.previewimg = None
        self.canvas.itemconfig(self.imgcanvas, image=self.previewimg)
