# Simple 'echo' example

This example demonstrates a very simple example that prints a message
to stdout.

``` yaml
echo:
  exec: echo ${message}
  flags:
    message:
      default: Hi there
```

To run the example, change to this directory and run:

    $ guild run echo

You may specify an optional `message` flag:

    $ guild run echo message='Hola mis amigos'

`exec` in this case runs the specified command. The value `${message}`
is used to include the `message` value as a command line argument to
the `echo` command.
