from __future__ import print_function

import argparse

import numpy as np
import tensorboardX

p = argparse.ArgumentParser()
p.add_argument("--x", type=float, default=1.0)
p.add_argument("--logs")

args = p.parse_args()

x = args.x

loss = (np.sin(5 * x) * (1 - np.tanh(x ** 2)) *
        np.random.randn() * 0.1)
print("loss: %.6f" % loss)

writer = tensorboardX.SummaryWriter(args.logs)
writer.add_scalar("x", x, 0)
writer.add_scalar("loss", loss, 0)
writer.close()
