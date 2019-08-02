# Noisy example

This example is from [Bayesian optimization with
skopt](https://scikit-optimize.github.io/notebooks/bayesian-optimization.html). It
uses a simple function to generate a simulated training loss. It can
be used to experiment with various hyperparameter search methods.

Here's a plot of the function's x (input) and loss (output):

![](https://scikit-optimize.github.io/notebooks/bayesian-optimization_files/bayesian-optimization_8_0.png)

The two scripts provided:

- [noisy.py](noisy.py) - Example using stdout to print loss
- [noisy2.py](noisy2.py) - Example using
  [tensorboardX](https://github.com/lanpa/tensorboardX) to log loss

To run the function with default values:

    $ guild run noisy.py

To run the function in a batch, use any of the following methods.

Grid search:

    $ guild run noisy.py x=[-1, 0, 1]

Random (5 runs):

    $ guild run noisy.py x=[-1.0:1.0] -m 5

Bayesian optimization (5 runs):

    $ guild run noisy.py x=[-2.0:2.0] -o bayesian -m 5

To compare results:

    $ guild compare

To view results in TensorBoard:

    $ guild tensorboard

Note that you can control the runs compared with either command by
specifying additional options:

- Run IDs or run ranges
- Operation names
- Run status

For more information, run a Guild command with the `--help` command
line option, e.g.:

    $ guild compare --help
