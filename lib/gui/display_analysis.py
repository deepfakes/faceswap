#!/usr/bin python3
""" Analysis tab of Display Frame of the Faceswap GUI """

import csv
import tkinter as tk
from tkinter import ttk

from .display_graph import SessionGraph
from .display_page import DisplayPage
from .stats import SavedSessions, SessionsSummary, QuartersSummary
from .tooltip import Tooltip
from .utils import Images, FileHandler

class Analysis(DisplayPage):
    """ Session analysis tab """
    def __init__(self, parent, tabname, helptext):
        DisplayPage.__init__(self, parent, tabname, helptext)

        self.loaded_data = None
        self.summary = None

        self.add_options()
        self.add_main_frame()

    def set_vars(self):
        """ Analysis specific vars """
        selected_id = tk.StringVar()
        selected_id.trace("w", self.update_session_summary)
        filename = tk.StringVar()
        return {'selected_id': selected_id,
                'filename': filename}

    def add_main_frame(self):
        """ Add the main frame to the subnotebook to hold stats and session data """
        mainframe = self.subnotebook_add_page('stats')
        self.stats = StatsData(mainframe,
                               self.vars['filename'],
                               self.vars['selected_id'],
                               self.helptext['stats'])
        self.sessionsummary = SessionSummary(mainframe, self.helptext['session'])

    def add_options(self):
        """ Add the options bar """
        self.reset_session_info()
        options = Options(self)
        options.add_options()

    def reset_session_info(self):
        """ Reset the session info status to default """
        self.vars['filename'].set(None)
        self.vars['info'].set('No session data loaded')

    def load_session(self):
        """ Load previously saved sessions """
        self.clear_session()
        filename = FileHandler('open', 'session').retfile
        if not filename:
            return
        filename = filename.name
        self.loaded_data = SavedSessions(filename).sessions
        self.summary = SessionsSummary(self.loaded_data).summary
        stattext = filename if len(filename) < 75 else '...{}'.format(filename[-75:])
        self.vars['info'].set('Session: {}'.format(stattext))
        self.vars['filename'].set(filename)
        self.stats.loaded_data = self.loaded_data
        self.stats.tree_insert_data(self.summary)

    def reset_session(self):
        """ Load previously saved sessions """
        if self.session.iterations == 0:
            print('Training not running')
            return
        self.clear_session()
        self.loaded_data = {key: value for key, value in self.session.historical.sessions.items()}
        self.loaded_data.update(self.session.compile_session())
        self.summary = SessionsSummary(self.loaded_data).summary
        self.vars['info'].set('Session: Currently running training session')
        self.stats.tree_insert_data(self.summary)

    def clear_session(self):
        """ Clear sessions stats """
        self.summary = None
        self.loaded_data = None
        self.stats.tree_clear()
        self.reset_session_info()

    def save_session(self):
        """ Save sessions stats to csv """
        if not self.summary:
            print('No summary data loaded. Nothing to save')
            return
        savefile = FileHandler('save', 'csv').retfile
        if not savefile:
            return

        write_dicts = [val for val in self.summary.values()]
        fieldnames = sorted(key for key in write_dicts[0].keys())

        with savefile as outfile:
            csvout = csv.DictWriter(outfile, fieldnames)
            csvout.writeheader()
            for row in write_dicts:
                csvout.writerow(row)

    def update_session_summary(self, *args):
        """ Update the session summary on receiving a value change
            callback from stats box """
        sessionid = self.vars['selected_id'].get()
        sessiondata = QuartersSummary(self.loaded_data, sessionid).summary
        self.sessionsummary.tree_insert_data(sessiondata)

class Options(object):
    """ Options bar of Analysis tab """
    def __init__(self, parent):
        self.optsframe = parent.optsframe
        self.parent = parent

    def add_options(self):
        """ Add the display tab options """
        self.add_buttons()

    def add_buttons(self):
        """ Add the option buttons """
        for btntype in ('reset', 'clear', 'save', 'load'):
            cmd = getattr(self.parent, '{}_session'.format(btntype))
            btn = ttk.Button(self.optsframe,
                             image=Images().icons[btntype],
                             command=cmd)
            btn.pack(padx=2, side=tk.RIGHT)
            hlp = self.set_help(btntype)
            Tooltip(btn, text=hlp, wraplength=200)

    @staticmethod
    def set_help(btntype):
        """ Set the helptext for option buttons """
        hlp = ""
        if btntype == 'reset':
            hlp = 'Load/Refresh stats for the currently training session'
        elif btntype == 'clear':
            hlp = 'Clear currently displayed session stats'
        elif btntype == 'save':
            hlp = 'Save session stats to csv'
        elif btntype == 'load':
            hlp = 'Load saved session stats'
        return hlp

class StatsData(ttk.Frame):
    """ Stats frame of analysis tab """
    def __init__(self, parent, filename, selected_id, helptext):
        ttk.Frame.__init__(self, parent)
        self.pack(side=tk.TOP, padx=5, pady=5, expand=True, fill=tk.X, anchor=tk.N)

        self.filename = filename
        self.loaded_data = None
        self.selected_id = selected_id
        self.popup_positions = list()

        self.add_label()
        self.tree = ttk.Treeview(self, height=1, selectmode=tk.BROWSE)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.columns = self.tree_configure(helptext)

    def add_label(self):
        """ Add Treeview Title """
        lbl = ttk.Label(self, text="Session Stats", anchor=tk.CENTER)
        lbl.pack(side=tk.TOP, expand=True, fill=tk.X, padx=5, pady=5)

    def tree_configure(self, helptext):
        """ Build a treeview widget to hold the sessions stats """
        self.tree.configure(yscrollcommand=self.scrollbar.set)
        self.tree.tag_configure('total', background='black', foreground='white')
        self.tree.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind('<ButtonRelease-1>', self.select_item)
        Tooltip(self.tree, text=helptext, wraplength=200)
        return self.tree_columns()

    def tree_columns(self):
        """ Add the columns to the totals treeview """
        columns = (("session", 40, '#'),
                   ("start", 130, None),
                   ("end", 130, None),
                   ("elapsed", 90, None),
                   ("batch", 50, None),
                   ("iterations", 90, None),
                   ("rate", 60, 'EGs/sec'))
        self.tree["columns"] = [column[0] for column in columns]

        for column in columns:
            text = column[2] if column[2] else column[0].title()
            self.tree.heading(column[0], text=text)
            self.tree.column(column[0], width=column[1], anchor=tk.E, minwidth=40)
        self.tree.column("#0", width=40)
        self.tree.heading("#0", text="Graphs")

        return [column[0] for column in columns]

    def tree_insert_data(self, sessions):
        """ Insert the data into the totals treeview """
        sessionids = sorted(key for key in sessions.keys())
        self.tree.configure(height=len(sessionids))

        for idx in sessionids:
            values = [sessions[idx][column] for column in self.columns]
            kwargs = {'values': values, 'image': Images().icons['graph']}
            if values[0] == 'Total':
                kwargs['tags'] = 'total'
            self.tree.insert('', 'end', **kwargs)

    def tree_clear(self):
        """ Clear the totals tree """
        self.tree.delete(* self.tree.get_children())
        self.tree.configure(height=1)

    def select_item(self, event):
        """ Update the session summary info with the selected item or launch graph """
        region = self.tree.identify("region", event.x, event.y)
        selection = self.tree.focus()
        values = self.tree.item(selection, 'values')
        if values:
            self.selected_id.set(values[0])
            if region == 'tree':
                self.data_popup()

    def data_popup(self):
        """ Pop up a window and control it's position """
        toplevel = SessionPopUp(self.loaded_data, self.selected_id.get())
        toplevel.title(self.data_popup_title())
        position = str(self.data_popup_get_position())
        toplevel.geometry('720x400+{}+{}'.format(position, position))
        toplevel.update()

    def data_popup_title(self):
        """ Set the data popup title """
        selected_id = self.selected_id.get()
        title = 'All Sessions' if selected_id == 'Total' else 'Session #{}'.format(selected_id)
        return '{} - {}'.format(title, self.filename.get())

    def data_popup_get_position(self):
        """ Get the position of the next window """
        # TODO Reset position when at edge of screen
        position = 120
        while True:
            if position not in self.popup_positions:
                self.popup_positions.append(position)
                break
            position += 20
        return position

class SessionSummary(ttk.Frame):
    """ Summary for selected session """
    # TODO Totals rate/times don't make sense because they are across sessions
    def __init__(self, parent, helptext):
        ttk.Frame.__init__(self, parent)
        self.pack(side=tk.BOTTOM, padx=5, pady=5, expand=True, fill=tk.X, anchor=tk.S)

        self.add_label()
        self.tree = ttk.Treeview(self, height=4, selectmode=tk.BROWSE)
        self.tree_configure(helptext)

        self.columns = self.tree_columns()

    def add_label(self):
        """ Add Treeview Title """
        lbl = ttk.Label(self, text="Individual Session Stats per Quarter", anchor=tk.CENTER)
        lbl.pack(side=tk.TOP, expand=True, fill=tk.X, padx=5, pady=5)

    def tree_configure(self, helptext):
        """ Pack treeview widget and add helptext """
        self.tree.pack(side=tk.TOP, expand=True, fill=tk.X)
        Tooltip(self.tree, text=helptext, wraplength=200)

    def tree_columns(self, loss_columns=None):
        """ Add the columns to the totals treeview """
        columns = [("quarter", 60, None),
                   ("start", 130, None),
                   ("end", 130, None),
                   ("elapsed", 90, None),
                   ("iterations", 90, None),
                   ("rate", 60, 'EGs/sec')]
        if loss_columns:
            for loss in loss_columns:
                columns.append((loss, 60, None))

        self.tree["columns"] = [column[0] for column in columns]
        self.tree['show'] = 'headings'

        for column in columns:
            text = column[2] if column[2] else column[0].title()
            self.tree.heading(column[0], text=text)
            self.tree.column(column[0], width=column[1], anchor=tk.E, minwidth=40, stretch=True)

        return [column[0] for column in columns]

    def tree_insert_data(self, selected_session):
        """ Insert the data into the session summary treeview """
        loss_columns = sorted([key for key in selected_session[1].keys()
                               if key not in self.columns])

        if loss_columns:
            self.tree_refresh_columns(loss_columns)

        self.tree_clear()
        quarters = sorted(key for key in selected_session.keys())
        for idx in quarters:
            values = [selected_session[idx][column] for column in self.columns]
            self.tree.insert('', 'end', values=values)

    def tree_refresh_columns(self, loss_columns):
        """ Refresh the treview with loss columns """
        self.tree.destroy()
        self.tree = ttk.Treeview(self, height=4, selectmode=tk.BROWSE)
        self.tree.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.columns = self.tree_columns(loss_columns)

    def tree_clear(self):
        """ Clear the totals tree """
        self.tree.delete(* self.tree.get_children())

class SessionPopUp(tk.Toplevel):
    """ Pop up for detailed grap/stats for selected session """
    def __init__(self, data, session_id):
        tk.Toplevel.__init__(self)

        self.graph = None

        self.vars = dict()
        self.graph_initialised = False
        self.build(data, session_id)

        for key, val in self.vars.items():
            print('{} - {}'.format(key, val.get()))

    def build(self, data, session_id):
        """ Build the popup window """
        optsframe, graphframe = self.layout_frames()

        self.opts_build(optsframe)
        self.graph_build(graphframe, data, session_id)

    def layout_frames(self):
        """ Top level container frames """
        leftframe = ttk.Frame(self)
        leftframe.pack(side=tk.LEFT, expand=False, fill=tk.BOTH, pady=5)

        sep = ttk.Frame(self, width=2, relief=tk.RIDGE)
        sep.pack(fill=tk.Y, side=tk.LEFT)

        rightframe = ttk.Frame(self)
        rightframe.pack(side=tk.RIGHT, fill=tk.BOTH, pady=5, expand=True)

        return leftframe, rightframe

    def opts_build(self, frame):
        """ Options in options to the optsframe """
        self.opts_combobox(frame)
        self.opts_checkbuttons(frame)
        self.opts_entry(frame)
        self.opts_buttons(frame)
        sep = ttk.Frame(frame, height=2, relief=tk.RIDGE)
        sep.pack(fill=tk.X, pady=(5, 0), side=tk.BOTTOM)

    def opts_combobox(self, frame):
        """ Add the options combo boxes """
        choices = {'Display': ('Loss', 'Rate'),
                   'Scale': ('Linear', 'Log')}

        for item in ['Display', 'Scale']:
            var = tk.StringVar()
            cmd = self.optbtn_reset if item == 'Display' else self.graph_scale
            var.trace("w", cmd)

            cmbframe = ttk.Frame(frame)
            cmbframe.pack(fill=tk.X, pady=5, padx=5, side=tk.TOP)
            lblcmb = ttk.Label(cmbframe, text='{}:'.format(item), width=7, anchor=tk.W)
            lblcmb.pack(padx=(0, 2), side=tk.LEFT)

            cmb = ttk.Combobox(cmbframe, textvariable=var, width=10)
            cmb['values'] = choices[item]
            cmb.current(0)
            cmb.pack(fill=tk.X, side=tk.RIGHT)

            self.vars[item.lower().strip()] = var

    def opts_checkbuttons(self, frame):
        """ Add the options check buttons """
        for text in ('Raw', 'Trend', 'Rolling Average'):
            var = tk.BooleanVar()
            if text == 'Raw':
                var.set(True)
            var.trace("w", self.optbtn_reset)
            self.vars[text.lower().replace(' ', '')] = var

            ctl = ttk.Checkbutton(frame, variable=var, text="Show {}".format(text))
            ctl.pack(side=tk.TOP, padx=5, pady=5, anchor=tk.W)

    def opts_entry(self, frame):
        """ Add the options entry boxes """
        entframe = ttk.Frame(frame)
        entframe.pack(fill=tk.X, pady=5, padx=5, side=tk.TOP)
        lblchk = ttk.Label(entframe, text="Iterations to Average:", anchor=tk.W)
        lblchk.pack(padx=(0, 2), side=tk.LEFT)

        ctl = ttk.Entry(entframe, width=4, justify=tk.RIGHT)
        ctl.pack(side=tk.RIGHT, anchor=tk.W)
        ctl.insert(0, '10')

        self.vars['avgiterations'] = ctl

    def opts_buttons(self, frame):
        """ Add the option buttons """
        btnframe = ttk.Frame(frame)
        btnframe.pack(fill=tk.X, pady=5, padx=5, side=tk.BOTTOM)

        for btntype in ('reset', 'save'):
            cmd = getattr(self, 'optbtn_{}'.format(btntype))
            btn = ttk.Button(btnframe,
                             image=Images().icons[btntype],
                             command=cmd)
            btn.pack(padx=2, side=tk.RIGHT)
            hlp = self.set_help(btntype)
            Tooltip(btn, text=hlp, wraplength=200)

    def optbtn_save(self):
        """ Action for clear button press """
        pass

    def optbtn_reset(self, *args):
        """ Action for reset button press and checkbox changes"""
        if not self.graph_initialised:
            return
        self.graph.refresh()

    def graph_scale(self, *args):
        """ Action for changing graph scale """
        if not self.graph_initialised:
            return
        self.graph.switch_yscale()

    @staticmethod
    def set_help(btntype):
        """ Set the helptext for option buttons """
        hlp = ""
        if btntype == 'reset':
            hlp = 'Refresh graph'
        elif btntype == 'save':
            hlp = 'Save graph'
        return hlp

    def graph_build(self, frame, data, session_id):
        """ Build the graph in the top right paned window """
        self.graph = SessionGraph(frame, self.vars, data, session_id)
        self.graph.pack(expand=True, fill=tk.BOTH)
        self.graph.build()
        self.graph_initialised = True
