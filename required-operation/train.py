import os

assert os.path.exists("data/data1.txt")
assert os.path.exists("data/subdir/data2.txt")

# Fake trained model!
open("model.json", "w").close()
open("checkpoint.h5", "w").close()
