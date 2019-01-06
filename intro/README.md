# Guild intro example

Using example from:

https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/19_Hyper-Parameters.ipynb

## Steps

High level:

- Install stuff
- Run train.py with Guild - show what happens
- Run train.py with --optimize - show what happens

Important points:

- Each trial is captured as a seperate experiment

- Trivial to reproduct results - good for collaboration and publishing

### Run train.py

Run the script:

    $ guild run train.py

Show the run, and the files.

## Changes needed in Guild

- Resource def local to op. Easy to do.
