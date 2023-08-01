## Simple install instruction

To build the library itself, follow these instructions.
These configure/make steps whill build the library. 

First build the npy_array library if you haven't done so earlier. `npy_array` is a separate project, but you will get
it in a separate folder if you did clone with the `--recursive` option. 

    $ cd npy_array
    $ ./configure
    $ make

If you do not have libzip installed on your system, it is recommended that you do. Your system probably has a package
manager that can install the package. On Ubuntu systems it is `sudo apt install libzip-dev`.

    $ cd src
    $ ./configure
    $ make

Note that the configure and makefile system is written from skratch and is not based on CMake, or GNU autoconf/automake.

You can now run through the examples in the `examples` directory.

    $ cd examples
    $ ./configure
    $ make

This will give you 3 example executable files that you can study further. Good luck.

