#!/usr/bin/env python3
"""
Source: http://home.wlu.edu/~levys/software/kbhit.py
A Python class implementing KBHIT, the standard keyboard-interrupt poller.
Works transparently on Windows and Posix (Linux, Mac OS X).  Doesn't work
with IDLE.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

import os
import sys

# Windows
if os.name == "nt":
    import msvcrt  # pylint:disable=import-error

# Posix (Linux, OS X)
else:
    import termios
    import atexit
    from select import select


class KBHit:
    """ Creates a KBHit object that you can call to do various keyboard things. """
    def __init__(self, is_gui=False):
        self.is_gui = is_gui
        if os.name == "nt" or self.is_gui or not sys.stdout.isatty():
            pass
        else:
            # Save the terminal settings
            self.file_desc = sys.stdin.fileno()
            self.new_term = termios.tcgetattr(self.file_desc)
            self.old_term = termios.tcgetattr(self.file_desc)

            # New terminal setting unbuffered
            self.new_term[3] = self.new_term[3] & ~termios.ICANON & ~termios.ECHO
            termios.tcsetattr(self.file_desc, termios.TCSAFLUSH, self.new_term)

            # Support normal-terminal reset at exit
            atexit.register(self.set_normal_term)

    def set_normal_term(self):
        """ Resets to normal terminal.  On Windows this is a no-op. """
        if os.name == "nt" or self.is_gui or not sys.stdout.isatty():
            pass
        else:
            termios.tcsetattr(self.file_desc, termios.TCSAFLUSH, self.old_term)

    def getch(self):
        """ Returns a keyboard character after kbhit() has been called.
            Should not be called in the same program as getarrow(). """
        if (self.is_gui or not sys.stdout.isatty()) and os.name != "nt":
            return None
        if os.name == "nt":
            return msvcrt.getch().decode("utf-8", errors="replace")
        return sys.stdin.read(1)

    def getarrow(self):
        """ Returns an arrow-key code after kbhit() has been called. Codes are
        0 : up
        1 : right
        2 : down
        3 : left
        Should not be called in the same program as getch(). """

        if (self.is_gui or not sys.stdout.isatty()) and os.name != "nt":
            return None
        if os.name == "nt":
            msvcrt.getch()  # skip 0xE0
            char = msvcrt.getch()
            vals = [72, 77, 80, 75]
        else:
            char = sys.stdin.read(3)[2]
            vals = [65, 67, 66, 68]

        return vals.index(ord(char.decode("utf-8", errors="replace")))

    def kbhit(self):
        """ Returns True if keyboard character was hit, False otherwise. """
        if (self.is_gui or not sys.stdout.isatty()) and os.name != "nt":
            return None
        if os.name == "nt":
            return msvcrt.kbhit()
        d_r, _, _ = select([sys.stdin], [], [], 0)
        return d_r != []
