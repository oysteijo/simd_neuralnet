## Simple install instruction

To build the library itself, follow these instructions.
These configure/make steps whill build the library. 

**Compiling npy_array**

First build the `npy_array` library if you haven't done so earlier. `npy_array` is a separate project, but added as a
subtree (not as a submodule). It in a separate folder. 
```shell
$ cd npy_array
$ ./configure
$ make
$ sudo make install
```
If you do not have libzip installed on your system, it is recommended that you do. Your system probably has a package
manager that can install the package. On Ubuntu systems it is `sudo apt install libzip-dev`, you may have to
do a `sudo apt update` + `sudo apt upgreade` first.

**Compiling SIMD NeuralNet**

```shell
$ cd src
$ ./configure
$ make
$ sudo make install
```
Note that the configure and makefile system is written from skratch and is not based on CMake, or GNU autoconf/automake.

You can now run through the examples in the `examples` directory.
```shell
$ cd examples
$ ./configure
$ make
```
This will give you 3 example executable files that you can study further. Good luck.

