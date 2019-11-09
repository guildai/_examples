import os

import tensorboardX

os.makedirs("a")

with tensorboardX.SummaryWriter("a") as writer:
    writer.add_scalar("val_acc", 0.5)
    writer.add_scalar("val/acc", 0.4)
    writer.add_scalar("acc/val", 0.3)
