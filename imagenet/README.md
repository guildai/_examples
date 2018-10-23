# ImageNet example

This example demonstrates the use of an external script [1] to implement
an operation.

The operation uses a wrapper to modify the external script behavior to
avoid re-downloading and unpacking the Inception pretrained model for each
operation without modifying the external script.

See [classify_image_wrapper.py](classify_image_wrapper.py), which uses
`guild.python_util` to execute the external script `classify_image.py`
(which is downloaded as an operation resource) by replacing the
`maybe_download_and_extract` function in `classify_image` with a
no-op.

The operation reuqires the `inception-model` resource, which handles
the download and unpacking of the model, but does so once and avoids
duplicating resource files across runs.

[1] https://raw.githubusercontent.com/tensorflow/models/master/tutorials/image/imagenet/classify_image.py
