

class Individual:
    """Individual.

    Base class for individuals that can be used in an Optimizer.
    """
    def __init__(self, logger=None):
        self.logger_ = logger

    def log(self, *args, log_level=1, id=0, **kwargs):
        if self.logger_ is not None:
            self.logger_.log(*args, log_level=log_level, id=id, **kwargs)

    def copy(self):
        """copy.
        
        Return copy of this object.
        """
        pass

    @classmethod
    def random(cls):
        """random.

        Return random individual.
        """
        pass

    def mutate(self, **kwargs):
        """mutate.

        Mutate in-place. Additional arguments may contain mutation rate etc.
        """
        pass

    def cross(self, other, **kwargs):
        """cross.

        Perform in-place crossover with `other`.

        other: Individual or list of Individuals to cross with.
        """
        pass
