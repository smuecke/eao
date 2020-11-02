import numpy as np


class Evaluator:
    """Evaluator.

    Base class that evaluates individuals. Must be implemented
    to match Individual implementation.
    """

    def __init__(self):
        pass

    def eval(self, individual):
        """eval.

        Evaluate individual, returning a numerical loss value.

        individual: Individual to evaluate.
        """
        if individual.loss is not None:
            return individual.loss
        else:
            raise NotImplementedError()
            # write loss value to individual

    def eval_all(self, individuals):
        """eval_all.

        Evaluate individuals in a given iterable, return numpy array of loss values.

        individuals: iterable containing Individuals.
        """
        return [ self.eval(ind) for ind in individuals ]
