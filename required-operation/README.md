# Required Operation Example

This example illustrates how an operation can require the output from
another operation.

Refer to the comments in [guild](guild.yml) for details.

To run the example, first try to run `train`:

    $ guild run train

The operation should fail with the message:

    Resolving prepared-data dependency
    guild: run failed because a dependency was not met: could
    not resolve 'operation:prepare-data' in prepared-data
    resource: no suitable run for prepare-data

This error indicates that `prepare-data` must be run first.

Let's do that:

    $ guild run prepare-data

This is a mock operation - it creates some empty files to simulate an
actual data prep operation.

View the prepare-data files:

    $ guild ls
    ~/.guild/runs/da39492a99614cbda3ed93500f9623ce:
      data1.txt
      subdir/
      subdir/data2.txt

Next, run train:

    $ guild run train

View the train files:

    $ guild ls
    ~/.guild/runs/f31be0c217b749ac8e3709813edd87a0:
      checkpoint.h5
      data/
      data/data1.txt
      data/subdir/
      model.json

The mock train files are `model.json` and `checkpoint.h5` (again, not
real). The other files are links to the files from the `prepare-data`
run. Note these files are located in a `data` subdirectory. This is
defined in [guild.yml](guild.yml) using the `path` attribute of the
operation requirement.

Next, run `train2`:

    $ guild run train2

And list the train2 files:

    $ guild ls
    ~/.guild/runs/f7b04e3e26d046ddb824fcea45874a05:
      checkpoint.h5
      data.txt
      model.json

Note that `data.txt` is the sole file from the `prepare-data`
operation. This is a renamed link to `subdir/data2.txt` from
`prepare-data`. Again, this is controlled by operation requirement in
[guild.yml](guild.yml).
