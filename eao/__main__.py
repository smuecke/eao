import numpy as np

from .individual import Individual
from .evaluator import Evaluator
from .optimizer import Optimizer
from .logger import Logger, LOG_VERBOSE

N = 10 # size of vector


class RealVector(Individual):

    def __init__(self, vec, logger=None):
        super().__init__(logger)
        self.vec = vec

    def __repr__(self):
        return str(self.vec) + " (loss={})".format(self.loss_)

    def copy(self):
        ind = self.__class__(self.vec.copy(), self.logger_)
        return ind

    @classmethod
    def random(cls, size, logger=None):
        return cls(np.random.normal(loc=0, scale=2, size=N), logger=logger)

    def mutate(self, mutation_rate=0.1, mutation_width=1):       
        if isinstance(mutation_width, int):
            # interpret as absolute number
            k = mutation_width
        elif isinstance(mutation_width, float):
            # interpret as probability per vector element
            k = max(1, np.random.binomial(self.vec.size, mutation_width))

        indices = np.random.choice(self.vec.size, size=k, replace=False)
        noise = np.random.normal(loc=0, scale=mutation_rate, size=k)
        self.vec[indices] += noise
        self.log('Added ' + ', '.join(['{} to vec[{}]'.format(noise_, ix) for ix, noise_ in zip(indices, noise)]) + 'of #'+str(self.id_), log_level=2, id=12)

    def cross(self, other, crossover_rate=0.5):
        # uniform crossover
        k = max(1, np.random.binomial(self.vec.size, crossover_rate))
        indices = np.random.choice(self.vec.size, size=k, replace=False)
        self.vec[indices] = other.vec[indices]
        self.log('#{} took indices {} from #{}'.format(self.id_, ', '.join(map(str,indices)), other.id_), log_level=2, id=13)


class RealVectorEvaluator(Evaluator):

    def __init__(self, target_vector, logger=None):
        super().__init__(logger)
        self.target_vec = target_vector

    def eval(self, ind):
        return np.linalg.norm(ind.vec - self.target_vec, 2)


def print_lines(ls):
    for l in ls:
        print(l)

conf = {
    'parents': 5,
    'offspring': 10,

    'do_crossover': True,
    'do_self_adaption': True,

    'mutation_prob': 0.9,
    'crossover_kwargs': {
        'crossover_rate': 0.5},
    'mutation_kwargs': {
        'mutation_rate': 1,
        'mutation_width': 0.1},

    'mutation_kwargs_bounds': {
        'mutation_width': (0.0, 1.0)
    }
}


with Logger(log_level=LOG_VERBOSE, file="main_log.txt") as l:
    ev = RealVectorEvaluator(np.random.uniform(-20, 20, size=N), logger=l)
    print('target: {}'.format(ev.target_vec))
    opt = Optimizer(ev, config=conf, logger=l)
    res = opt.run([RealVector.random(N, logger=l) for _ in range(5)], generations=50)

#print_lines(res)
