import argparse

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--batch-size", type=int, default=100)

args = p.parse_args()

print(
    "Training %s with learning rate of %f and batch size of %i"
    % (args.model, args.lr, args.batch_size))
