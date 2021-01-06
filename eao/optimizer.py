import numpy as np

from .config         import default_config
from .logging        import logger
from datetime        import datetime
from itertools       import count
from multiprocessing import Pool
from sys             import stdout
from tqdm            import trange


def coinflip(p=0.5):
    return np.random.binomial(1, p) == 1

def logistic(x):
    return 1/(1+np.exp(-x))

def logit(x):
    return np.log(x/(1-x))

def logit_perturb(val, valmin, valmax, learning_rate):
    valspan = valmax-valmin
    val_ = logit((val - valmin) / valspan) # normalize and transform
    val_ += np.random.normal(loc=0, scale=learning_rate) # add noise
    return valspan * logistic(val_) + valmin # transform back and unnormalize


class Optimizer:

    def __init__(self, evaluator, config=None):
        self.evaluator = evaluator

        # use defaults and overwrite with custom options
        self.config = default_config.copy()
        if config is not None:
            self.config.update(config)

        self.__check_config()

    def __check_config(self):
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
    
    def __mutate_parameters(self, ind):
        """mutate_parameters.

        Perform mutation on the EA parameters themselves, using a
        logit-normal distributed update.
        
        ind: Individual whose parameters to mutate.
        """
        for op in ['mutation', 'crossover']:
            # only mutate those parameters for which there are bounds given
            for arg, bounds in self.config[op + '_kwargs_bounds'].items():
                valmin, valmax = sorted(bounds)
                val = ind.config_[op + '_kwargs'][arg] # current value
                val_ = logit_perturb(val, valmin, valmax, self.config['learning_rate'])
                ind.config_[op + '_kwargs'][arg] = val_

    def run(self, initial, generations=10):
        logger.log('Starting run on {}'.format(datetime.now().strftime("%c")), level=1, id=0)
        logger.log('Configuration is {}; running for {} generations'.format(', '.join([k+'='+str(v) for k,v in self.config.items()]), generations), level=1, id=1)
        id_counter = count(start=0, step=1)

        npar = self.config['parents']
        noff = self.config['offspring']

        # initialize population
        if len(initial) != npar:
            raise ValueError("given initial population size {} does not match parent population size {}".format(len(initial), npar))
        
        parents = []
        for ind in initial:
            ind_ = ind.copy()
            ind_.loss_ = None
            ind_.id_ = next(id_counter)
            if self.config['do_self_adaption']:
                ind_.config_ = {
                    'mutation_kwargs': self.config['mutation_kwargs'].copy(),
                    'crossover_kwargs': self.config['crossover_kwargs'].copy()
                }
            parents.append(ind_)
        
        if not self.config['do_self_adaption']:
            ind_config = self.config

        # initially evaluate parents
        parent_loss = self.evaluator.eval_all(parents)
        logger.log("Parents have loss {}".format(', '.join(['#{}={}'.format(p.id_, p.loss_) for p in parents])), level=1, id=2)

        # if log is not None:
        #     logfile = open(log, 'w')
        #     logfile.write("t,{}\n".format(','.join(["l"+str(i) for i in range(npar)])))
        #     logfile.write("0,{}\n".format(','.join([str(p.loss_) for p in parents])))
        #     previous_loss = parent_loss[:]

        # initially sort parents
        parents.sort(key=lambda ind: ind.loss_)
        logger.log("Sorting initial parent population by loss", level=2, id=3)

        offspring = []
        for generation in trange(generations):
            logger.log('Entering generation {}'.format(generation), level=1, id=4)

            # sample random parent indices
            parent_ixs = np.random.randint(0, npar, size=noff)

            for ix in parent_ixs:
                ind = parents[ix].copy()
                ind.id_ = next(id_counter)
                logger.log('Copied #{} to new offspring #{}'.format(parents[ix].id_, ind.id_), level=2, id=5)

                if self.config['do_self_adaption']:
                    ind.config_ = parents[ix].config_.copy()
                    self.__mutate_parameters(ind)
                    ind_config = ind.config_ # use already mutated parameters
                    logger.log('Mutated #{}\'s parameters to {}'.format(ind.id_, ', '.join([k+'='+str(v) for k,v in ind.config_.items()])), level=2, id=6)

                if self.config['do_crossover'] and coinflip(self.config['crossover_prob']):
                    other_ind = parents[(ix + np.random.randint(1, npar)) % npar]
                    logger.log('Crossing #{} with #{}'.format(ind.id_, other_ind.id_), level=2, id=7)
                    ind.cross(other_ind, **ind_config['crossover_kwargs'])
                if self.config['do_mutate'] and coinflip(self.config['mutation_prob']):
                    logger.log('Mutating #{}'.format(ind.id_), level=2, id=8)
                    ind.mutate(**ind_config['mutation_kwargs'])

                ind.loss_ = None # invalidate loss (just in case)
                offspring.append(ind)

            # evaluate offspring and sort by loss
            self.evaluator.eval_all(offspring)
            logger.log("Offspring have loss {}".format(', '.join(['#{}={}'.format(ind.id_, ind.loss_) for ind in offspring])), level=1, id=9)

            offspring.sort(key=lambda ind: ind.loss_)
            logger.log("Sorting offspring population by loss", level=2, id=10)

            if self.config['selection'] == 'plus':
                # perform plus selection:
                # sort better offspring into parent population,
                # preferring offspring when loss is equal;
                # this uses the fact that parents and offspring are internally sorted
                pix, oix = 0, 0
                while (pix < npar) and (oix < noff):
                    if parents[pix].loss_ >= offspring[oix].loss_:
                        logger.log('offspring #{} replaces parent #{}'.format(offspring[oix].id_, parents[pix].id_), level=2, id=11)
                        parents.insert(pix, offspring[oix].copy())
                        del parents[-1]
                        parents[pix].id_ = offspring[oix].id_
                        parents[pix].loss_ = offspring[oix].loss_
                        if self.config['do_self_adaption']:
                            parents[pix].config_ = offspring[oix].config_
                        oix += 1
                    pix += 1
            
            elif self.config['selection'] == 'comma':
                # perform comma selection:
                # select best offspring and discard previous parents
                for i, ind in enumerate(offspring[:npar]):
                    parents[i] = ind.copy()
                    parents[i].id_ = ind.id_
                    parents[i].loss_ = ind.loss_

            else:
                raise ValueError('Somehow you managed to slip in an unknown selection type. Have you messed with configuration checking?')

            offspring.clear()
            logger.log("Parents have loss {}".format(', '.join(['#{}={}'.format(p.id_, p.loss_) for p in parents])), level=1, id=2)

        return parents
