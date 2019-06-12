# Operation labels

This example demonstrates custom labels for operations.

Refer to [`guild.yml`](guild.yml) for examples.

To see the generated labels in a batch run, use:

```
$ guild run custom-label a=[1,2,3] -y
$ guild run disable-label a=[1,2,3] -y
$ guild run default-label a=[1,2,3] -y
```

To view the runs and their labels:

```
$ guild runs
```
