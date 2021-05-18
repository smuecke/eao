# Evolutionary Algorithm Optimization (EAO)

Package for optimization using Evolutionary Algorithms (EA), providing base classes for individuals and an evaluation module.


## Version History

* **0.4** Removed necessity to manually assign `loss_` attribute
* **0.5** Added extensive logging capabilities; will be cleaned up a little in later versions
* **0.6** Fixed critical bug that affects optimization with more than one parent
* **0.7** Made logging much cleaner; added plus-selection


## Documentation

`eao` (Evolutionary Algorithm Optimization)

This library provides a function minimizer using a simple modular evolutionary algorithm scheme that can be adapted to all kinds of target domains.
The algorithm performs optimization steps (generations) on a population of solutions by iteratively creating offspring through recombination and mutation, evaluation the offspring using a customizable fitness function and selecting new parents for the next generation, either through *plus* selection (best individuals from offspring plus previous parents) or *comma* selection (best individuals from offspring only).
Additionally, `eao` has useful logging capabilities, that allow to fully retrace each optimization run for further analysis.

In its core, `eao` requires the user to extend 2 classes:

* `Individual`: Class representing the target domain, e.g. real numbers, vectors, bit strings or more complex objects
* `Evaluator`: Class containing the evaluation method, which takes an object of `Individual` and assigns it a *loss value* (or inverse fitness value) according to which it is ranked

The following sections will explain how to implement these classes and how to use them with the optimizer.
As an example, we will implement an optimization problem on real vectors.

### Individual

In a first step, we will create a subclass of `Individual` called `RealVector` that will represent vectors of float values with fixed size. I'll be using `numpy` arrays as the actual internal data structure and write a wrapper that implements the methods required for individuals.
For a minimal implementation we will need the following methods:

`copy()`
: This method returns a (deep) copy of the object. This is necessary because objects are copied between populations internally, and this method allows us to manually copy internal class attributes that require manual deep copying on their own, such as `numpy` arrays.

`random()`
: This is a static method (`@classmethod`) and returns a random instance of our data type

`mutate(*kwargs)`
: This method performs *mutation* in-place, i.e. it introduces a little variation to the individual. For real vectors, we will add some random noise to one of the elements. Additional parameters can be passed to this function, which can be set via the configuration dict, more on this later.

`cross(other, *kwargs)`
: This performs *crossover* (recombination) in-place, taking another individual as input. For our example, we will perform *one-point crossover*, meaning we'll sample a cutting point and take one part from either parent. Just as `mutate`, this method can take custom parameters. Also, if you don't want to use crossover, you can skip this method.

After implementing these methods for our example application, the `RealVector` class looks like this:

```python
import numpy as np
from eao import Individual, Evaluator, Optimizer

class RealVector(Individual):

    def __init__(self, vec):
        self.vec = vec

    def copy(self):
        return RealVector(self.vec.copy())

    @classmethod
    def random(cls):
        return cls(np.random.uniform(-1, 1, size=16))

    def mutate(self):
        index = np.random.choice(16)
        self.vec[index] += np.random.normal(loc=0, scale=0.1)

    def cross(self, other):
        index = np.random.choice(17)
        self.vec[ix:] = other.vec[ix:]
```

As you can see we'll be working with RealVectors of size 16, and random instances will have elements uniformly distributed between -1 and 1.
When implementing `random`, make sure the initial individuals are not unreasonable, e.g. much too large or too small - this will make it difficult to reach the solution by small mutations.

### Evaluator

The `Evaluator` class contains the loss function that we'll have to specify for our optimization problem.
We only need to implement a single method:

```python
class RealVectorEvaluator(Evaluator):

    def eval(self, rv):
        even_indices = rv[::2]
        odd_indices = rv[1::2]
        return np.sum((even_indices-1)**2) + np.sum((odd_indices+1)**2)
```

For our example, we'll be be adding all even indices' distance to 1 and all odd indices' distance to -1, so that our global optimum is a vector of alternating 1 and -1. This nonsensical loss function is purely for demonstration purposes.

### Configuration

* Coming soon!


