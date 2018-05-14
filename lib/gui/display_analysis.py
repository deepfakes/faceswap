#!/usr/bin python3
""" Analysis tab of Display Frame of the Faceswap GUI """

import csv
import tkinter as tk
from tkinter import ttk

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
        var = tk.StringVar()
        var.trace("w", self.update_session_summary)
        return {'selected_id': var}

    def add_main_frame(self):
        """ Add the main frame to the subnotebook to hold stats and session data """
        mainframe = self.subnotebook_add_page('stats')
        self.stats = StatsData(mainframe, self.vars['selected_id'], self.helptext['stats'])
        self.sessionsummary = SessionSummary(mainframe, self.helptext['session'])

    def add_options(self):
        """ Add the options bar """
        self.reset_session_info()
        options = Options(self)
        options.add_options()

    def reset_session_info(self):
        """ Reset the session info status to default """
        self.vars['info'].set('No session data loaded')

    def load_session(self):
        """ Load previously saved sessions """
        filename = FileHandler('open', 'session').retfile.name
        if not filename:
            return
        self.clear_session()
        self.loaded_data = SavedSessions(filename).sessions
        self.summary = SessionsSummary(self.loaded_data).summary
        stattext = filename if len(filename) < 75 else '...{}'.format(filename[-75:])
        self.vars['info'].set('Session: {}'.format(stattext))
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
    def __init__(self, parent, selected_id, helptext):
        ttk.Frame.__init__(self, parent)
        self.pack(side=tk.TOP, padx=5, pady=5, expand=True, fill=tk.X, anchor=tk.N)

        self.add_label()
        self.selected_id = selected_id
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

    def select_item(self, arg):
        """ Update the session summary info with the selected item """
        selection = self.tree.focus()
        values = self.tree.item(selection, 'values')
        if values:
            self.selected_id.set(values[0])

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
