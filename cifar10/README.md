# CIFAR10

This is a [Guild AI](http://guild.ai) example that defines a CIFAR10
model based on TensorFlow's excellent [Convolutional Neural
Networks](https://www.tensorflow.org/tutorials/deep_cnn/) Guide.

## Project files

- **[Guild](Guild)** - Guild project file ([more info](https://guild.ai/project-reference/))
- **[samples.py](samples.py)** - Implementation of the *samples* resource
- **[single.py](expert.py)** - Training script for single GPU
- **[support.py](support.py)** - Shared code across training scripts
- **[upstream_single.py](upstream_single.py)** - Wrapper to train
  using the original TensorFlow (upstream) script for single GPU

## Requirements

To work with this project, [ensure that you have Guild AI installed along
with its requirements](https://guild.ai/getting-started/setup/).

If you haven't already, clone Guild AI examples:

```
$ git clone https://github.com/guildai/guild-examples.git
```

## Training

Prepare and train the CIFAR10 model:

``` bash
$ cd guild-examples/cifar10
$ guild prepare
$ guild train
```

The `prepare` command downloads the CIFAR10 data, which will be used
for subsequent operations.

## Viewing

At any point you can view project run results by in [Guild
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
$ guild serve RUN
```

where `RUN` is a value returned by `list-runs` (i.e. a path to the run
directory) or `--latest-run` to serve the model exported in the last
run.

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
