#!/usr/bin python3
""" Graph functions for Display Frame of the Faceswap GUI """
import datetime
import os
import tkinter as tk

from tkinter import ttk
from math import ceil, floor

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt, style
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import numpy

from .stats import PopUpData
from .tooltip import Tooltip
from .utils import Images


class NavigationToolbar(NavigationToolbar2Tk):
    """ Same as default, but only including buttons we need
        with custom icons and layout
        From: https://stackoverflow.com/questions/12695678 """
    toolitems = [t for t in NavigationToolbar2Tk.toolitems if
                 t[0] in ('Home', 'Pan', 'Zoom', 'Save')]

    @staticmethod
    def _Button(self, text, file, command, extension='.gif'):
        """ Same as original byt Custom button icons,
            ttk used and packed to right """
        iconmapping = {'home': 'reset',
                       'filesave': 'save',
                       'zoom_to_rect': 'zoom'}
        icon = iconmapping[file] if iconmapping.get(file, None) else file
        img = Images().icons[icon]
        btn = ttk.Button(self, text=text, image=img, command=command)
        btn.pack(side=tk.RIGHT, padx=2)
        return btn

    def _init_toolbar(self):
        """ Same as original but ttk widgets and standard
            tooltips used. Separator added and message label
            packed to the left """
        xmin, xmax = self.canvas.figure.bbox.intervalx
        height, width = 50, xmax-xmin
        ttk.Frame.__init__(self, master=self.window,
                           width=int(width), height=int(height))

        sep = ttk.Frame(self, height=2, relief=tk.RIDGE)
        sep.pack(fill=tk.X, pady=(5, 0), side=tk.TOP)

        self.update()  # Make axes menu

        btnframe = ttk.Frame(self)
        btnframe.pack(fill=tk.X, padx=5, pady=5, side=tk.RIGHT)

        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                # Add a spacer; return value is unused.
                self._Spacer()
            else:
                button = self._Button(btnframe, text=text, file=image_file,
                                      command=getattr(self, callback))
                if tooltip_text is not None:
                    Tooltip(button, text=tooltip_text, wraplength=200)

        self.message = tk.StringVar(master=self)
        self._message_label = ttk.Label(master=self, textvariable=self.message)
        self._message_label.pack(side=tk.LEFT, padx=5)
        self.pack(side=tk.BOTTOM, fill=tk.X)

class GraphBase(ttk.Frame):
    """ Base class for matplotlib line graphs """
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        style.use('ggplot')

        self.colourmaps = ['Reds', 'Blues', 'Greens',
                           'Purples', 'Oranges', 'Greys',
                           'copper', 'summer', 'bone']
        self.toolbar = None
        self.fig = plt.figure(figsize=(4, 4), dpi=75)
        self.ax1 = self.fig.add_subplot(1, 1, 1)
        self.plotcanvas = FigureCanvasTkAgg(self.fig, self)

        self.plotcanvas.get_tk_widget().pack(side=tk.TOP, padx=5, fill=tk.BOTH, expand=True)
        plt.subplots_adjust(left=0.100, bottom=0.100, right=0.95, top=0.95,
                            wspace=0.2, hspace=0.2)

    def axes_labels_set(self, ylabel):
        """ Set the axes label and range """
        self.ax1.set_xlabel('Iterations')
        self.ax1.set_ylabel(ylabel)

    def axes_limits_set_default(self):
        """ Set default axes limits """
        self.ax1.set_ylim(0.00, 100.0)
        self.ax1.set_xlim(0, 1)

    def axes_limits_set(self, xmax, data):
        """ Set the axes limits """
        if data:
            ymin, ymax = self.axes_data_get_min_max(data)
            self.ax1.set_ylim(ymin, ymax)
            self.ax1.set_xlim(0, xmax)
        else:
            self.axes_limits_set_default()

    @staticmethod
    def axes_data_get_min_max(data):
        """ Return the minimum and maximum values from list of lists """
        ymin, ymax = list(), list()
        for item in data:
            dataset = list(filter(lambda x: x is not None, item))
            ymin.append(min(dataset) * 1000)
            ymax.append(max(dataset) * 1000)
        ymin = floor(min(ymin)) / 1000
        ymax = ceil(max(ymax)) / 1000
        return ymin, ymax

    def axes_set_yscale(self, scale):
        """ Set the Y-Scale to log or linear """
        self.ax1.set_yscale(scale)

    def lines_sort(self, keys):
        """ Sort the data keys into consistent order
            and set line colourmap and line width """
        raw_lines = list()
        sorted_lines = list()
        for key in sorted(keys):
            title = key.replace('_', ' ').title()
            if key.startswith(('avg', 'trend')):
                sorted_lines.append([key, title])
            else:
                raw_lines.append([key, title])

        groupsize = self.lines_groupsize(raw_lines, sorted_lines)
        sorted_lines = raw_lines + sorted_lines

        lines = self.lines_style(sorted_lines, groupsize)
        return lines

    @staticmethod
    def lines_groupsize(raw_lines, sorted_lines):
        """ Get the number of items in each group.
            If raw data isn't selected, then check
            the length of remaining groups until
            something is found """
        groupsize = 1
        if raw_lines:
            groupsize = len(raw_lines)
        else:
            for check in ('avg', 'trend'):
                if any(item[0].startswith(check) for item in sorted_lines):
                    groupsize = len([item for item in sorted_lines if item[0].startswith(check)])
                    break
        return groupsize

    def lines_style(self, lines, groupsize):
        """ Set the colourmap and linewidth for each group """
        groups = int(len(lines) / groupsize)
        colours = self.lines_create_colors(groupsize, groups)
        for idx, item in enumerate(lines):
            linewidth = ceil((idx + 1) / groupsize)
            item.extend((linewidth, colours[idx]))
        return lines

    def lines_create_colors(self, groupsize, groups):
        """ Create the colours """
        colours = list()
        for i in range(1, groups + 1):
            for colour in self.colourmaps[0:groupsize]:
                cmap = matplotlib.cm.get_cmap(colour)
                cpoint = 1 - (i / 5)
                colours.append(cmap(cpoint))
        return colours

    def legend_place(self):
        """ Place and format legend """
        self.ax1.legend(loc='upper right', ncol=2)

    def toolbar_place(self, parent):
        """ Add Graph Navigation toolbar """
        self.toolbar = NavigationToolbar(self.plotcanvas, parent)
        self.toolbar.pack(side=tk.BOTTOM)
        self.toolbar.update()


class TrainingGraph(GraphBase):
    """ Live graph to be displayed during training. """

    def __init__(self, parent, loss, losskeys):
        GraphBase.__init__(self, parent)

        self.loss = (losskeys, loss)
        self.anim = None
        self.lines = {'losslines': list(), 'trndlines': list()}

    def build(self):
        """ Update the plot area with loss values and cycle through to
        animate """
        self.axes_labels_set("Loss")
        self.axes_limits_set_default()

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
        self.legend_place()
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
        xlim = len(loss[0])
        xlim = 2 if xlim == 1 else xlim

        self.axes_limits_set(xlim - 1, loss)

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
        #TODO Move trendplot to stats.
        #TODO Standardise stats between live and session
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

class SessionGraph(GraphBase):
    """ Session Graph for session pop-up """
    def __init__(self, parent, tkvars, data, session_id):
        GraphBase.__init__(self, parent)
        self.compiled = PopUpData(data, tkvars, session_id)
        self.display = tkvars['display']
        self.scale = tkvars['scale']

    def build(self):
        """ Build the session graph """
        self.update_plot()
        self.toolbar_place(self)
        self.plotcanvas.draw()

    def update_plot(self):
        """ Update the plot area """
        self.ax1.clear()
        self.axes_labels_set(self.display.get())

        fulldata = [item for item in self.compiled.data.values()]
        self.axes_limits_set(self.compiled.iterations, fulldata)

        xrng = [x for x in range(self.compiled.iterations)]
        keys = list(self.compiled.data.keys())
        for item in self.lines_sort(keys):
            self.ax1.plot(xrng,
                          self.compiled.data[item[0]],
                          label=item[1],
                          linewidth=item[2],
                          color=item[3])
        self.legend_place()

    def refresh(self):
        """ Refresh graph data """
        self.compiled.refresh()
        self.update_plot()
        self.plotcanvas.draw()

    def switch_yscale(self):
        """ switch the y-scale and redraw """
        self.update_plot()
        self.axes_set_yscale(self.scale.get())
        self.plotcanvas.draw()
