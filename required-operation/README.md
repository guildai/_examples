# Required Operation Example

This example illustrates how an operation can require the output from
another operation.

Refer to the comments in [`guild.yml`](guild.yml) for additional
information.

## Basics

To run the example, first try to run `train`:

    $ guild run train

The operation should fail with the message:

    Resolving prepared-data dependency
    guild: run failed because a dependency was not met: could not resolve
    'operation:prepare-data' in prepared-data resource: no suitable run for
    prepare-data

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

The mock train files are `model.json` and `checkpoint.h5` (both
empty). The other files are links to the files from the `prepare-data`
run. Note these files are located in a `data` subdirectory. This is
defined in [`guild.yml`](guild.yml) using the `path` attribute of the
operation requirement.

You can show the dependencies for a run by including the `-d,
--dependencies` option when running `guild runs info`:

    $ guild runs info -d
    id: f31be0c217b749ac8e3709813edd87a0
    <snip>
    dependencies:
      prepared-data:
        ~/.guild/runs/da39492a99614cbda3ed93500f9623ce/data1.txt
        ~/.guild/runs/da39492a99614cbda3ed93500f9623ce/subdir

The files listed under `dependencies` above show the paths to the
sources used from the prepare-data run.

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
[`guild.yml`](guild.yml).

Show the dependencies for the train2 run:

    $ guild runs info -d
    id: f7b04e3e26d046ddb824fcea45874a05
    <snip>
    dependencies:
      prepared-data:
        ~/.guild/runs/da39492a99614cbda3ed93500f9623ce/subdir/data2.txt

## Specify a run for a requirement

By default Guild uses the latest non-error run for a required
operation. You can specify an alternative run in one of two ways:

- Mark the run you want to use using the `mark` command

- Specify the run ID of the operation you want using flag syntax for
  `prepared-data`

An explicit run ID takes precedence over any other method.

To illustrate, run `prepare-data` a second time to so that there are
multiple `prepare-data` runs:

    $ guild run prepare-data

Confirm you have mutiple prepare-data runs:

    $ guild runs -o prepare-data
    [1:e3afaf34]  prepare-data  2019-07-30 12:58:33  completed
    [2:da39492a]  prepare-data  2019-07-30 12:44:00  completed

By default, new runs of `train` or `train2` will use the second run
for `prepare-data` (run `e3afaf34` in the example above - your run IDs
will be different). To use the first `prepare-data` run, specify its
run ID as follows:

     $ guild run train prepared-data=da39492a

Note again, your run IDs will be different. Replace `da39492a` above
with the applicable `prepare-data` run ID on your system.

Note the resolved dependency in the train output:

    Using output from run da39492a99614cbda3ed93500f9623ce for prepared-data
    resource

If there's a run that you want to use implicitly, use the `mark`
command:

    $ guild mark da39492a

Replace `da39492a` with the applicable run ID on your system.

When you run `train` or `train2` without specifying a value for
`prepared-data` the marked run is used.

    $ guild run train

Note again the run ID displayed in the output.

If you have multiple marked runs, Guild uses the latest marked run of
the required operation. You can clear a mark later using `guild mark
--clear RUN-ID`.