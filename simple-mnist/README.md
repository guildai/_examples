# Simple MNIST example

This illustrates a very simple yet functional example of using Guild
to run experiments.

A generic Python script using TensorFlow:

    $ guild run train.py epochs=5

A Keras script:

    $ guild run mnist_mlp.py epochs=5

If the script doesn't write its files to the current working
directory, then it's not going to work, or the user would have to run
it this way:

    $ guild run train.py run_dir=.

Under the covers, if we see a command like this, we could assume this
Guild file:

``` yaml
model: ''
operations:
  train.py: train
```

We'd also implicitly set the --force-flag option to avoid complaints
about unrecognized flags.
