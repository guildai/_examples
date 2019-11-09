# Compare Columns

The columns that appear when comparing a run in `guild compare` or
`guild view` can defined explicitly via the `compare` operation attr.

The syntax is:

```
compare:
  - KEY [ 'as' COLNAME ]
  ...
```

See [guild.yml](guild.yml) for an example.

To recreate, create a virtual environment (e.g. by running `guild
init` in this directory or using Conda or virtualenv). Activate the
environment and install the packages in `requirements.txt` by running
`pip install -r requirements.txt` (if you used `guild init` these are
automatically installed).

With the activate env, run:

```
$ guild run test -y
```

View the runs in compare:

```
$ guild compare -t
```
