# TF Lite example

This example demonstrates how support for [TensorFlow
Lite](https://www.tensorflow.org/lite/) can be added to a project.  In
this example, we create a MobileNet v2 classifier for pets with
`gpkg.slim` and add support for generating a TF Lite file with
`gpkg.tflite`.

## Run the example

Clone examples repos:

    $ git clone https://github.com/guildai/examples.git

Change to the `tflite` example and create a Guild environment (the
environment is used to isolate runs and installed packages):

    $ cd examples/tflite
    $ guild init

NOTE: If you want to run this example using the latest Guild packages,
ensure that you have cloned the [packages
repo](https://github.com/guildai/packages) and initialize the
environment with `-p`, providing the path to the local packages repo:

    $ guild init -p $PACKAGES

where `PACKAGES` is the path to the local cloned packages repo.

The project tests will run the sequence of operations needed to
generate a tflite file. You can run the tests to demonstrate model
features:

    $ guild test

Alternatively, if you want to train the example to optimal accuracy,
run the sequence beflow.

Prepare the images for training:

    $ guild run prepare

Use transfer learning from a model trained on ImageNet:

    $ guild run transfer-learn train-steps=5000

Export and freeze the inference graph:

    $ guild run export-and-freeze

Finally, generate a `tflite` file:

    $ guild run tflite

For help in using the `tflite` file in a mobile application, see *[Use
the TensorFlow Lite model for inference in a mobile
app](https://www.tensorflow.org/lite/devguide#3_use_the_tensorflow_lite_model_for_inference_in_a_mobile_app)*.
