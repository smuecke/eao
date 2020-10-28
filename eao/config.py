default_config = {
    
    # population sizes
    'parents': 1,
    'offspring': 10,
    
    # enable/disable mutation and crossover
    'do_mutate': True,
    'do_crossover': False,

    # set mutation/crossover parameters
    'crossover_prob': 0.5,
    'mutation_prob': 1.0,

    # additional arguments
    'crossover_kwargs': {},
    'mutation_kwargs': {},
    
    'do_timeout_heuristic': False,
    'timeout': 100,
    'mutation_increment': 1.01,
    'max_mutation_rate': 0.98,

    # selection
    'selection': 'plus', # options: ['plus', 'comma']

    # multiprocessing options
    'kernels': 1
}
