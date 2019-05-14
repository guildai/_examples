import sys

a = 1
b = 2

print("step: 1")
print("loss: 1.123")
print("accuracy: 0.123")

# Stderr works
sys.stderr.write("mse: 0.1231\n")
