AUTOMAKE_OPTIONS = foreign
noinst_LIBRARIES = libneuralnet.a
libneuralnet_a_SOURCES = neuralnet.c activation.c matrix_multiply.c loss.c
libneuralnet_a_CFLAGS = $(AVX2_FLAGS) $(INCLUDE_PATH) $(DEFINES)

bin_PROGRAMS = example
example_SOURCES = example.c
example_CFLAGS = $(AVX2_FLAGS) $(INCLUDE_PATH) $(DEFINES)
example_LDADD = libneuralnet.a -L../c_npy -lc_npy -lcblas -lm
