# MNIST

This is a [Guild AI](http://guild.ai) example that provides two MNIST
models:

- **intro** - logistic regression
- **expert** - convolutional neural network

The terms *intro* and *expert* &mdash; along with the corresponding models
themselves &mdash; are from TensorFlow's excellent
[Introduction](https://www.tensorflow.org/get_started/).

## Project files

- **[expert.py](expert.py)** - Implementation of the *expert* model
- **[Guild](Guild)** - Guild project file ([more info](https://guild.ai/project-reference/))
- **[intro.py](intro.py)** - Implementation of the *intro* model
- **[samples.py](samples.py)** - Implementation of the *samples* resource

## Requirements

To work with these models, [ensure that you have Guild AI installed along
with its requirements](https://guild.ai/getting-started/setup/).

If you haven't already, clone Guild AI examples:

```
$ git clone https://github.com/guildai/guild-examples.git
```

## Training

Prepare and train the MNIST model:

``` bash
$ cd guild-examples/mnist
$ guild prepare
$ guild train
```

The `prepare` command downloads the MNIST data, which will be used for
all subsequent trainin.

This project is configured to train the `intro` model by trained. You
can train `expert` by specifying it explicitly:

``` bash
# Alternatively train the expert model
$ guild train expert
```

## Viewing

At any point you can project run results by in [Guild
View](https://guild.ai/user-guide/guild-view) by running the
`view` command in a separate terminal:

``` bash
$ guild view
```

Guild View runs on port `6333` by default &mdash; to view it,
open [http://localhost:6333](http://localhost:6333) in your browser.

## Evaluating

To calculate the most recently trained model's final accuracy using
test data, use the `evaluate` command:

``` bash
$ guild evaluate --latest-run
```

## Serving

To run a model as an HTTP service in [Guild
Serve](https://guild.ai/user-guide/guild-serve/) run:

``` bash
$ guild serve
```

## Generating sample inputs

This example provides a `samples` resource that can be used to
generate a number of MNIST images and their corresponding JSON
encodings. These file can be used to run ad hoc inference on the model
in Guild View (see the **Serve** tab) or Guild Serve
(see [Serving](#user-content-serving) above).

Images are generated from the MNIST training/test data.

To generate samples, run:

``` bash
$ guild prepare samples
```

This will create a local `samples` subdirectory containing the samples
images.

## More about Guild AI

For more information about the Guild AI project, see
[https://guild.ai](https://guild.ai).
