import os
import sys
import contextlib

class suppress_stdout_stderr(object):
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')
        
        self.old_stdout_fileno_undup    = sys.stdout.fileno()
        self.old_stdout_fileno = os.dup ( sys.stdout.fileno() )
        self.old_stdout = sys.stdout        
        os.dup2 ( self.outnull_file.fileno(), self.old_stdout_fileno_undup )
        sys.stdout = self.outnull_file
  
        self.old_stderr_fileno_undup    = sys.stderr.fileno()
        self.old_stderr_fileno = os.dup ( sys.stderr.fileno() )
        self.old_stderr = sys.stderr        
        os.dup2 ( self.errnull_file.fileno(), self.old_stderr_fileno_undup )
        sys.stderr = self.errnull_file
        return self
        
    def __exit__(self, *_):
        os.dup2 ( self.old_stdout_fileno, self.old_stdout_fileno_undup )
        sys.stdout = self.old_stdout
        
        os.dup2 ( self.old_stderr_fileno, self.old_stderr_fileno_undup )
        sys.stderr = self.old_stderr        

        self.outnull_file.close()
        self.errnull_file.close()

        os.close ( self.old_stdout_fileno )
        os.close ( self.old_stderr_fileno )