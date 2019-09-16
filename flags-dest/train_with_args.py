import argparse  # cue for Guild that interface is `args`

p = argparse.ArgumentParser()
p.add_argument("--learning-rate", type=float, default=0.01)
p.add_argument("--epochs", type=int, default=10)

args = p.parse_args()

print("Training for %i epochs with a learning rate of %f"
      % (args.epochs, args.learning_rate))
