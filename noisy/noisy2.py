import numpy as np

import tensorboardX as tbx

x = 0.1
noise = 0.1

def f(x):
    return np.sin(5 * x) * (1 - np.tanh(x ** 2)) + np.random.randn() * noise

loss = f(x)

writer = tbx.SummaryWriter(".")
writer.add_scalar("loss", loss)
writer.close()
