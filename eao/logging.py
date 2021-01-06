from datetime import datetime
from io       import TextIOWrapper
from sys      import stdout


LOG_SILENT  = 0
LOG_NORMAL  = 1
LOG_VERBOSE = 2


class Logger():

    def __init__(self, level=LOG_NORMAL, output=stdout):
        self.__level = level
        self.set_output(output)

    def set_output(self, output):
        try:
            if self.__closable:
                self.__out.close()
        except AttributeError:
            # no output open
            pass

        if isinstance(output, str):
            self.__out = open(output, 'w')
            self.__closable = True
        elif isinstance(output, TextIOWrapper): 
            self.__out = output
            self.__closable = output.name not in ['<stdout>', '<stderr>']

    def close(self):
        if self.__closable:
            self.__out.close()

    def set_level(self, level):
        if level in [0, 1, 2]:
            self.__level = level
        else:
            raise ValueError('Invalid log level: must be 0 (silent), 1 (normal) or 2 (verbose)')

    def log(self, *msg, level=LOG_NORMAL, id=0, **kwargs):
        if level <= self.__level and not self.__out.closed:
            timestamp = datetime.now().strftime("[%H:%M:%S.%f]")
            print(timestamp, '('+str(id)+')', *msg, **kwargs, file=self.__out, flush=True)


# singleton instance
logger = Logger()
