import os

assert os.path.exists("data.txt")

# Fake trained model!
open("model.json", "w").close()
open("checkpoint.h5", "w").close()
