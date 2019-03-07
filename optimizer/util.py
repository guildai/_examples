def save_model_summary(model, filename='model.txt'):
    with open(filename, 'w') as f:
        model.summary(print_fn=lambda out: f.write(out + '\n'))
