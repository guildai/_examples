# Flags

Flags are parameter inputs to scripts. Common flags include *learning
rate*, *batch size*, and *training epochs* but flags may be used to
convey any type of information to a script.

Guild lets you run scripts with different flag values. For example, to
specify a value for `learning-rate` when running a script `train.py`,
use:

```
$ guild run train.py learning-rate=0.01
```

Guild interfaces with a script in one of two ways:

- Command line arguments
- Global variables

Guild automatically detects which mode it should use by examining your
script for the use of Python's `argparse` module. If your script
imports `argparse`, Guild assumes the *arg* mode. If your script does
not import `argparse`, Guild assumes *globals* mode.

You can specify the mode in a Guild file using the `flags-dest`
attribute for the applicable operation. When you specify this value in
a Guild file, Guild does not inspect your script or make any
assumptions about how to set flags for your script.

This example project illustrate the two
modes. [train_with_globals.py](train_with_globals.py) uses *globals*
mode and [train_with_args.py](train_with_args.py) uses *args*
mode. Either script can be run directly.

Using global variables:

```
$ guild run train_with_globals.py
You are about to run train_with_globals.py
  epochs: 10
  learning_rate: 0.01
Continue? (Y/n) y
Training for 10 epochs with a learning rate of 0.010000
```

Using command line arguments:

```
$ guild run train_with_args.py
You are about to run train_with_args.py
  epochs: 10
  learning_rate: 0.01
Continue? (Y/n) y
Training for 10 epochs with a learning rate of 0.010000
```

In both cases, Guild inspects the script and decides which mode to use
based on whether or not the script imports `argparse`.

The Guild file [`guild.yml`](guild.yml) defines two operations:
`train-with-globals` and `train-with-args`, which explicitly set
`flags-dest`.

The Guild file also defines a third operation `train-with-click`,
implemented by [train_with_click.py](train_with_click.py), which uses
[Click](https://click.palletsprojects.com) to parse command line
arguments. In this case, the target module does not import `argparse`
and so Guild's default behavior is incorrect. The configuration in
`guild.yml` tells Guild to set flag values as command line arguments
and not as global variables. Additionally, because Guild does not
auto-detect flags from scripts that use Click, you must define flags
in the Guild file.

Note that you cannot run `train_with_click.py` as a script because
Guild does not support Click-based interfaces. You must run the
operation `train-with-click`, which provides the interface details.

Therefore, this command will NOT detect flags in the script:

```
$ guild run train_with_click.py`
```

This command uses information from the Guild file:

```
$ guild run train-with-click
```
