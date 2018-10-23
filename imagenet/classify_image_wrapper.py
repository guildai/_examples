from guild import python_util

# Skip maybe_download_and_extract - we always expect the model to be
# available under --model_dir location.

python_util.exec_script(
    "classify_image.py",
    { "maybe_download_and_extract": lambda: None })
