from datetime import datetime
from io       import TextIOWrapper
from sys      import stdout

LOG_SILENT  = 0
LOG_NORMAL  = 1
LOG_VERBOSE = 2

class Logger():

    def __init__(self, log_level=LOG_NORMAL, file=stdout, flush_immediately=False, close_on_exit=False):
        self.log_level = log_level
        self.file = file
        self.flush_immediately = flush_immediately
        self.close_on_exit = close_on_exit

    def __enter__(self):
        if isinstance(self.file, str):
            # treat as filename
            self.__file = open(self.file, 'w')

        elif isinstance(self.file, TextIOWrapper):
            # treat as file (e.g. for stdout)
            self.__file = self.file

        return self

    def __exit__(self, type, value, traceback):
        if self.close_on_exit:
            self.__file.close()

    def log(self, *msg, log_level=LOG_NORMAL, id=0, **kwargs):
        if log_level <= self.log_level and not self.__file.closed:
            timestamp = datetime.now().strftime("[%H:%M:%S.%f]")
            print(timestamp, '('+str(id)+')', *msg, **kwargs, file=self.__file, flush=self.flush_immediately)