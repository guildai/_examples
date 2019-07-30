import os

# Fake data FTW!

open("data1.txt", "w").close()

os.mkdir("subdir")
open("subdir/data2.txt", "w").close()
