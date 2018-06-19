#!/usr/bin python3
""" Analysis tab of Display Frame of the Faceswap GUI """

import csv
import tkinter as tk
from tkinter import ttk

from .display_graph import SessionGraph
from .display_page import DisplayPage
from .stats import Calculations, SavedSessions, SessionsSummary, SessionsTotals
from .tooltip import Tooltip
from .utils import Images, FileHandler


class Analysis(DisplayPage):
    """ Session analysis tab """
    def __init__(self, parent, tabname, helptext, scaling_factor):
        DisplayPage.__init__(self, parent, tabname, helptext)

        self.summary = None
        self.add_options()
        self.add_main_frame(scaling_factor)

    def set_vars(self):
        """ Analysis specific vars """
        selected_id = tk.StringVar()
        filename = tk.StringVar()
        return {"selected_id": selected_id,
                "filename": filename}

    def add_main_frame(self, scaling_factor):
        """ Add the main frame to the subnotebook
            to hold stats and session data """
        mainframe = self.subnotebook_add_page("stats")
        self.stats = StatsData(mainframe,
                               self.vars["filename"],
                               self.vars["selected_id"],
                               self.helptext["stats"],
                               scaling_factor)

    def add_options(self):
        """ Add the options bar """
        self.reset_session_info()
        options = Options(self)
        options.add_options()

    def reset_session_info(self):
        """ Reset the session info status to default """
        self.vars["filename"].set(None)
        self.set_info("No session data loaded")

    def load_session(self):
        """ Load previously saved sessions """
        self.clear_session()
        filename = FileHandler("open", "session").retfile
        if not filename:
            return
        filename = filename.name
        loaded_data = SavedSessions(filename).sessions
        msg = filename
        if len(filename) > 70:
            msg = "...{}".format(filename[-70:])
        self.set_session_summary(loaded_data, msg)
        self.vars["filename"].set(filename)

    def reset_session(self):
        """ Load previously saved sessions """
        self.clear_session()
        if self.session.stats["iterations"] == 0:
            print("Training not running")
            return
        loaded_data = self.session.historical.sessions
        msg = "Currently running training session"
        self.set_session_summary(loaded_data, msg)
        self.vars["filename"].set("Currently running training session")

    def set_session_summary(self, data, message):
        """ Set the summary data and info message """
        self.summary = SessionsSummary(data).summary
        self.set_info("Session: {}".format(message))
        self.stats.loaded_data = data
        self.stats.tree_insert_data(self.summary)

    def clear_session(self):
        """ Clear sessions stats """
        self.summary = None
        self.stats.loaded_data = None
        self.stats.tree_clear()
        self.reset_session_info()

    def save_session(self):
        """ Save sessions stats to csv """
        if not self.summary:
            print("No summary data loaded. Nothing to save")
            return
        savefile = FileHandler("save", "csv").retfile
        if not savefile:
            return

        write_dicts = [val for val in self.summary.values()]
        fieldnames = sorted(key for key in write_dicts[0].keys())

        with savefile as outfile:
            csvout = csv.DictWriter(outfile, fieldnames)
            csvout.writeheader()
            for row in write_dicts:
                csvout.writerow(row)


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
        for btntype in ("reset", "clear", "save", "load"):
            cmd = getattr(self.parent, "{}_session".format(btntype))
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
        if btntype == "reset":
            hlp = "Load/Refresh stats for the currently training session"
        elif btntype == "clear":
            hlp = "Clear currently displayed session stats"
        elif btntype == "save":
            hlp = "Save session stats to csv"
        elif btntype == "load":
            hlp = "Load saved session stats"
        return hlp


class StatsData(ttk.Frame):
    """ Stats frame of analysis tab """
    def __init__(self,
                 parent,
                 filename,
                 selected_id,
                 helptext,
                 scaling_factor):
        ttk.Frame.__init__(self, parent)
        self.pack(side=tk.TOP,
                  padx=5,
                  pady=5,
                  expand=True,
                  fill=tk.X,
                  anchor=tk.N)

        self.filename = filename
        self.loaded_data = None
        self.selected_id = selected_id
        self.popup_positions = list()
        self.scaling_factor = scaling_factor

        self.add_label()
        self.tree = ttk.Treeview(self, height=1, selectmode=tk.BROWSE)
        self.scrollbar = ttk.Scrollbar(self,
                                       orient="vertical",
                                       command=self.tree.yview)
        self.columns = self.tree_configure(helptext)

    def add_label(self):
        """ Add Treeview Title """
        lbl = ttk.Label(self, text="Session Stats", anchor=tk.CENTER)
        lbl.pack(side=tk.TOP, expand=True, fill=tk.X, padx=5, pady=5)

    def tree_configure(self, helptext):
        """ Build a treeview widget to hold the sessions stats """
        self.tree.configure(yscrollcommand=self.scrollbar.set)
        self.tree.tag_configure("total",
                                background="black",
                                foreground="white")
        self.tree.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<ButtonRelease-1>", self.select_item)
        Tooltip(self.tree, text=helptext, wraplength=200)
        return self.tree_columns()

    def tree_columns(self):
        """ Add the columns to the totals treeview """
        columns = (("session", 40, "#"),
                   ("start", 130, None),
                   ("end", 130, None),
                   ("elapsed", 90, None),
                   ("batch", 50, None),
                   ("iterations", 90, None),
                   ("rate", 60, "EGs/sec"))
        self.tree["columns"] = [column[0] for column in columns]

        for column in columns:
            text = column[2] if column[2] else column[0].title()
            self.tree.heading(column[0], text=text)
            self.tree.column(column[0],
                             width=column[1],
                             anchor=tk.E,
                             minwidth=40)
        self.tree.column("#0", width=40)
        self.tree.heading("#0", text="Graphs")

        return [column[0] for column in columns]

    def tree_insert_data(self, sessions):
        """ Insert the data into the totals treeview """
        self.tree.configure(height=len(sessions))

        for item in sessions:
            values = [item[column] for column in self.columns]
            kwargs = {"values": values, "image": Images().icons["graph"]}
            if values[0] == "Total":
                kwargs["tags"] = "total"
            self.tree.insert("", "end", **kwargs)

    def tree_clear(self):
        """ Clear the totals tree """
        self.tree.delete(* self.tree.get_children())
        self.tree.configure(height=1)

    def select_item(self, event):
        """ Update the session summary info with
            the selected item or launch graph """
        region = self.tree.identify("region", event.x, event.y)
        selection = self.tree.focus()
        values = self.tree.item(selection, "values")
        if values:
            self.selected_id.set(values[0])
            if region == "tree":
                self.data_popup()

    def data_popup(self):
        """ Pop up a window and control it's position """
        toplevel = SessionPopUp(self.loaded_data, self.selected_id.get())
        toplevel.title(self.data_popup_title())
        position = self.data_popup_get_position()
        height = int(720 * self.scaling_factor)
        width = int(400 * self.scaling_factor)
        toplevel.geometry("{}x{}+{}+{}".format(str(height),
                                               str(width),
                                               str(position[0]),
                                               str(position[1])))
        toplevel.update()

    def data_popup_title(self):
        """ Set the data popup title """
        selected_id = self.selected_id.get()
        title = "All Sessions"
        if selected_id != "Total":
            title = "Session #{}".format(selected_id)
        return "{} - {}".format(title, self.filename.get())

    def data_popup_get_position(self):
        """ Get the position of the next window """
        init_pos = [120, 120]
        pos = init_pos
        while True:
            if pos not in self.popup_positions:
                self.popup_positions.append(pos)
                break
            pos = [item + 200 for item in pos]
            init_pos, pos = self.data_popup_check_boundaries(init_pos, pos)
        return pos

    def data_popup_check_boundaries(self, initial_position, position):
        """ Check that the popup remains within the screen boundaries """
        boundary_x = self.winfo_screenwidth() - 120
        boundary_y = self.winfo_screenheight() - 120
        if position[0] >= boundary_x or position[1] >= boundary_y:
            initial_position = [initial_position[0] + 50, initial_position[1]]
            position = initial_position
        return initial_position, position


class SessionPopUp(tk.Toplevel):
    """ Pop up for detailed grap/stats for selected session """
    def __init__(self, data, session_id):
        tk.Toplevel.__init__(self)

        self.is_totals = True if session_id == "Total" else False
        self.data = self.set_session_data(data, session_id)

        self.graph = None
        self.display_data = None

        self.vars = dict()
        self.graph_initialised = False
        self.build()

    def set_session_data(self, sessions, session_id):
        """ Set the correct list index based on the passed in session is """
        if self.is_totals:
            data = SessionsTotals(sessions).stats
        else:
            data = sessions[int(session_id) - 1]
        return data

    def build(self):
        """ Build the popup window """
        optsframe, graphframe = self.layout_frames()

        self.opts_build(optsframe)
        self.compile_display_data()
        self.graph_build(graphframe)

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
        choices = {"Display": ("Loss", "Rate"),
                   "Scale": ("Linear", "Log")}

        for item in ["Display", "Scale"]:
            var = tk.StringVar()
            cmd = self.optbtn_reset if item == "Display" else self.graph_scale
            var.trace("w", cmd)

            cmbframe = ttk.Frame(frame)
            cmbframe.pack(fill=tk.X, pady=5, padx=5, side=tk.TOP)
            lblcmb = ttk.Label(cmbframe,
                               text="{}:".format(item),
                               width=7,
                               anchor=tk.W)
            lblcmb.pack(padx=(0, 2), side=tk.LEFT)

            cmb = ttk.Combobox(cmbframe, textvariable=var, width=10)
            cmb["values"] = choices[item]
            cmb.current(0)
            cmb.pack(fill=tk.X, side=tk.RIGHT)

            self.vars[item.lower().strip()] = var

            hlp = self.set_help(item)
            Tooltip(cmbframe, text=hlp, wraplength=200)

    def opts_checkbuttons(self, frame):
        """ Add the options check buttons """
        for item in ("raw", "trend", "avg", "outliers"):
            if item == "avg":
                text = "Show Rolling Average"
            elif item == "outliers":
                text = "Flatten Outliers"
            else:
                text = "Show {}".format(item.title())
            var = tk.BooleanVar()

            if item == "raw":
                var.set(True)
            var.trace("w", self.optbtn_reset)
            self.vars[item] = var

            ctl = ttk.Checkbutton(frame, variable=var, text=text)
            ctl.pack(side=tk.TOP, padx=5, pady=5, anchor=tk.W)

            hlp = self.set_help(item)
            Tooltip(ctl, text=hlp, wraplength=200)

    def opts_entry(self, frame):
        """ Add the options entry boxes """
        for item in ("avgiterations", ):
            if item == "avgiterations":
                text = "Iterations to Average:"
                default = "10"

            entframe = ttk.Frame(frame)
            entframe.pack(fill=tk.X, pady=5, padx=5, side=tk.TOP)
            lbl = ttk.Label(entframe, text=text, anchor=tk.W)
            lbl.pack(padx=(0, 2), side=tk.LEFT)

            ctl = ttk.Entry(entframe, width=4, justify=tk.RIGHT)
            ctl.pack(side=tk.RIGHT, anchor=tk.W)
            ctl.insert(0, default)

            hlp = self.set_help(item)
            Tooltip(entframe, text=hlp, wraplength=200)

            self.vars[item] = ctl

    def opts_buttons(self, frame):
        """ Add the option buttons """
        btnframe = ttk.Frame(frame)
        btnframe.pack(fill=tk.X, pady=5, padx=5, side=tk.BOTTOM)

        for btntype in ("reset", "save"):
            cmd = getattr(self, "optbtn_{}".format(btntype))
            btn = ttk.Button(btnframe,
                             image=Images().icons[btntype],
                             command=cmd)
            btn.pack(padx=2, side=tk.RIGHT)
            hlp = self.set_help(btntype)
            Tooltip(btn, text=hlp, wraplength=200)

    def optbtn_save(self):
        """ Action for save button press """
        savefile = FileHandler("save", "csv").retfile
        if not savefile:
            return

        save_data = self.display_data.stats
        fieldnames = sorted(key for key in save_data.keys())

        with savefile as outfile:
            csvout = csv.writer(outfile, delimiter=",")
            csvout.writerow(fieldnames)
            csvout.writerows(zip(*[save_data[key] for key in fieldnames]))

    def optbtn_reset(self, *args):
        """ Action for reset button press and checkbox changes"""
        if not self.graph_initialised:
            return
        self.compile_display_data()
        self.graph.refresh(self.display_data,
                           self.vars["display"].get(),
                           self.vars["scale"].get())

    def graph_scale(self, *args):
        """ Action for changing graph scale """
        if not self.graph_initialised:
            return
        self.graph.set_yscale_type(self.vars["scale"].get())

    @staticmethod
    def set_help(control):
        """ Set the helptext for option buttons """
        hlp = ""
        control = control.lower()
        if control == "reset":
            hlp = "Refresh graph"
        elif control == "save":
            hlp = "Save display data to csv"
        elif control == "avgiterations":
            hlp = "Number of data points to sample for rolling average"
        elif control == "outliers":
            hlp = "Flatten data points that fall more than 1 standard " \
                  "deviation from the mean to the mean value."
        elif control == "avg":
            hlp = "Display rolling average of the data"
        elif control == "raw":
            hlp = "Display raw data"
        elif control == "trend":
            hlp = "Display polynormal data trend"
        elif control == "display":
            hlp = "Set the data to display"
        elif control == "scale":
            hlp = "Change y-axis scale"
        return hlp

    def compile_display_data(self):
        """ Compile the data to be displayed """
        self.display_data = Calculations(self.data,
                                         self.vars["display"].get(),
                                         self.selections_to_list(),
                                         self.vars["avgiterations"].get(),
                                         self.vars["outliers"].get(),
                                         self.is_totals)

    def selections_to_list(self):
        """ Compile checkbox selections to list """
        selections = list()
        for key, val in self.vars.items():
            if (isinstance(val, tk.BooleanVar)
                    and key != "outliers"
                    and val.get()):
                selections.append(key)
        return selections

    def graph_build(self, frame):
        """ Build the graph in the top right paned window """
        self.graph = SessionGraph(frame,
                                  self.display_data,
                                  self.vars["display"].get(),
                                  self.vars["scale"].get())
        self.graph.pack(expand=True, fill=tk.BOTH)
        self.graph.build()
        self.graph_initialised = True
