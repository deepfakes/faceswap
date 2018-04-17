import os
import sys

class suppress_stdout_stderr(object):
    def __init__(self):
        # Open a pair of null files
        self.stdout_fileno = sys.stdout.fileno()
        self.stderr_fileno = sys.stderr.fileno()
        
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(self.stdout_fileno), os.dup(self.stderr_fileno)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],self.stdout_fileno)
        os.dup2(self.null_fds[1],self.stderr_fileno)
        
    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],self.stdout_fileno)
        os.dup2(self.save_fds[1],self.stderr_fileno)
        
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
