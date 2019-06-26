clang -std=gnu99 -I.. -I../../neuralnet -I../../c_npy -Wall -Wextra -Wno-initializer-overrides -O3 -g -mavx2 -mfma -c test_sgd.c
clang -o test_sgd test_sgd.o -L../ -loptimizers -L../../neuralnet -lneuralnet -L../../c_npy -lc_npy -lomp -lm
