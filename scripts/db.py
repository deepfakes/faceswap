import cv2

from pathlib import Path
from tqdm import tqdm
import os

from lib import db
from lib.cli import DirectoryProcessor

class DBManager(DirectoryProcessor):
    def create_parser(self, subparser, command, description):
        self.parser = subparser.add_parser(
            command,
            help="Manage Sqlite3 Database.",
            description=description,
            )

    def add_optional_arguments(self, parser):
        parser.add_argument('-f', '--flush',
                            action='store_true',
                            dest='db_flush',
                            default=False,
                            help="Clears database tables.")
        
        parser.add_argument('-d', '--delete',
                            action="store_true",
                            dest="db_delete",
                            default=False,
                            help="Deletes database schema.")
        return parser

    def process(self):
        conn = db.open_connection()

        if self.arguments.db_flush:
            print("Flushing Database")
            db.flush(self.conn)
            print("Done")

        if self.arguments.db_delete:
            print("Deleting Database")
            db.delete(self.conn)
            print("Done")

