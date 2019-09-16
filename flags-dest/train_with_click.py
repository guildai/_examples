try:
    import click
except ImportError:
    raise SystemExit("click is required - try 'pip install click'")

@click.command()
@click.option("--learning-rate", default=0.01)
@click.option("--epochs", default=10)
def train(learning_rate, epochs):
    print("Training for %i epochs with a learning rate of %f"
          % (epochs, learning_rate))

train()
