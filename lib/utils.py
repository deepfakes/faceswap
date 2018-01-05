import argparse

from pathlib import Path
from scandir import scandir

def get_folder(path):
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def ensure_file_exists(dir, filename):
    file_path = Path(dir) / filename
    if not file_path.exists():
        print("File {} does not exist, creating...".format(file_path))
        file_path.touch()
    return file_path

def get_image_paths(directory):
    return [x.path for x in scandir(directory) if x.name.endswith('.jpg') or x.name.endswith('.jpeg') or x.name.endswith('.png')]

class FullHelpArgumentParser(argparse.ArgumentParser):
    """
    Identical to the built-in argument parser, but on error
    it prints full help message instead of just usage information
    """
    def error(self, message):
        self.print_help(sys.stderr)
        args = {'prog': self.prog, 'message': message}
        self.exit(2, '%(prog)s: error: %(message)s\n' % args)