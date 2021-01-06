class Individual:
    """Individual.

    Base class for individuals that can be used in an Optimizer.
    """
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
