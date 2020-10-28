import numpy as np

from .config         import default_config
from multiprocessing import Pool
from tqdm            import trange


def coinflip(p=0.5):
    return np.random.binomial(1, p) == 1


class Optimizer:

    def __init__(self, evaluator, config=None):
        self.evaluator = evaluator

        # use defaults and overwrite with custom options
        self.config = default_config.copy()
        if config is not None:
            self.config.update(config)
            
        self.check_config()

    def check_config(self):
        """check_config.
        
        Check if the configuration is valid, and raise error if there are any
        inconsistencies or impossible combinations.
        """
        if not (1 <= self.config['parents'] <= self.config['offspring']):
            raise ValueError('`parents` and `offspring` must be positive integers, with `offspring` being greater or equal to `parents`')
        
        if (self.config['parents'] == 1) and self.config['do_crossover']:
            raise ValueError('crossover can only be performed with at least 2 `parents`')
        
        if self.config['selection'] not in ['plus', 'comma']:
            raise ValueError('unknown selection {}, must be \'plus\' or \'comma\''.format(self.config['selection']))

    def run(self, initial, generations=10):

        npar = self.config['parents']
        noff = self.config['offspring']

        # initialize population
        if len(initial) != npar:
            raise ValueError("given initial population size {} does not match parent population size {}".format(len(initial), npar))
        parents = [ ind.copy() for ind in initial ]
        
        # initially evaluate parents
        self.evaluator.eval_all(parents)
        
        # initially sort parents
        parents.sort(key=lambda ind: ind.loss)
        #parent_loss, parents = zip(*sorted(zip(parent_loss, parents)))
        
                
        def augment(args):
            ix, ind = args
            if self.config['do_crossover'] and coinflip(self.config['crossover_prob']):
                ind.cross(parents[(ix + np.random.randint(1, npar)) % npar], **self.config['crossover_kwargs'])
            if self.config['do_mutate'] and coinflip(self.config['mutation_prob']):
                ind.mutate(**self.config['mutation_kwargs'])
            return ind
        
        offspring = []
        for generation in trange(generations):
            
            # sample random parent indices
            parent_ixs = np.random.randint(0, npar, size=noff)
            
            for ix in parent_ixs:
                ind = parents[ix].copy()
                
                if self.config['do_crossover'] and coinflip(self.config['crossover_prob']):
                    ind.cross(parents[(ix + np.random.randint(1, npar)) % npar], **self.config['crossover_kwargs'])
                if self.config['do_mutate'] and coinflip(self.config['mutation_prob']):
                    ind.mutate(**self.config['mutation_kwargs'])

                offspring.append(ind)

            # evaluate offspring and sort by loss
            self.evaluator.eval_all(offspring)
            offspring.sort(key=lambda ind: ind.loss)
            #offspring_loss, offspring = zip(*sorted(zip(self.evaluator.eval_all(offspring), offspring)))

            # sort better offspring into parent population,
            # preferring offspring when loss is equal;
            # this uses the fact that parents and offspring are internally sorted
            pix, oix = 0, 0
            while (pix < npar) and (oix < noff):
                if parents[pix].loss >= offspring[oix].loss:
                    parents[pix] = offspring[oix].copy()
                    oix += 1
                pix += 1
                
            offspring.clear()
            
        return parents
