import numpy as np

x = 0.1
noise = 0.1
loss = (np.sin(5 * x) * (1 - np.tanh(x ** 2)) * np.random.randn() * noise)

print("x: %f" % x)
print("noise: %f" % noise)
print("loss: %f" % loss)
