import numpy as np

from .individual import Individual
from .evaluator import Evaluator
from .optimizer import Optimizer


N = 10 # size of vector


class RealVector(Individual):

    def __init__(self, vec):
        super().__init__()
        self.vec = vec

    def __repr__(self):
        return str(self.vec) + " (loss={})".format(self.loss)

    def copy(self):
        ind = self.__class__(self.vec.copy())
        ind.loss = self.loss
        return ind

    @classmethod
    def random(cls, size):
        return cls(np.random.normal(loc=0, scale=2, size=N))

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
        self.loss = None

    def cross(self, other, crossover_rate=0.5):
        # uniform crossover
        k = max(1, np.random.binomial(self.vec.size, crossover_rate))
        indices = np.random.choice(self.vec.size, size=k, replace=False)
        self.vec[indices] = other.vec[indices]
        self.loss = None


class RealVectorEvaluator(Evaluator):

    def __init__(self, target_vector):
        self.target_vec = target_vector

    def eval(self, ind):
        if ind.loss is not None:
            return ind.loss
        else:
            loss = np.linalg.norm(ind.vec - self.target_vec, 2)
            ind.loss = loss
            return loss


def print_lines(ls):
    for l in ls:
        print(l)

conf = {
    'parents': 5,
    'offspring': 10,
    'do_crossover': True,
    'mutation_prob': 0.9,
    'crossover_kwargs': {
        'crossover_rate': 0.5},
    'mutation_kwargs': {
        'mutation_rate': 1,
        'mutation_width': 0.1}
}

ev = RealVectorEvaluator(np.random.uniform(-20, 20, size=N))
print('target: {}'.format(ev.target_vec))
opt = Optimizer(ev, config=conf)
res = opt.run([RealVector.random(N) for _ in range(5)], generations=1000, log="opt_log.csv")
print_lines(res)
