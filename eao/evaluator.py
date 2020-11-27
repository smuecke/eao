import numpy as np


class Evaluator:
    """Evaluator.

    Base class that evaluates individuals. Must be implemented
    to match Individual implementation.
    """

    def __init__(self, logger=None):
        self.logger_ = logger

    def log(self, *args, log_level=1, id=0, **kwargs):
        if self.logger_ is not None:
            self.logger_.log(*args, log_level=log_level, id=id, **kwargs)

    def eval(self, individual):
        """eval.

        Evaluate individual, returning a numerical loss value.

        individual: Individual to evaluate.
        """
        raise NotImplementedError()

    def eval_all(self, individuals):
        """eval_all.

        Evaluate individuals in a given iterable, return list of loss values.

        individuals: iterable containing Individuals.
        """
        losses = []
        for ind in individuals:
            if ind.loss_ is None:
                loss = self.eval(ind)
                ind.loss_ = loss
            losses.append(ind.loss_)
        return losses
