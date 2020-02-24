# FIXME: Make a loop or something
clang -std=gnu99 -I.. -I../../neuralnet -I../../c_npy -Wall -Wextra -Wno-initializer-overrides -O3 -g -mavx2 -mfma -c test_sgd.c
clang -o test_sgd test_sgd.o -L../ -loptimizers -L../../neuralnet -lneuralnet -L../../c_npy -lc_npy -lomp -lm

clang -std=gnu99 -I.. -I../../neuralnet -I../../c_npy -Wall -Wextra -Wno-initializer-overrides -O3 -g -mavx2 -mfma -c test_adagrad.c
clang -o test_adagrad test_adagrad.o -L../ -loptimizers -L../../neuralnet -lneuralnet -L../../c_npy -lc_npy -lomp -lm

clang -std=gnu99 -I.. -I../../neuralnet -I../../c_npy -Wall -Wextra -Wno-initializer-overrides -O3 -g -mavx2 -mfma -c test_rmsprop.c
clang -o test_rmsprop test_rmsprop.o -L../ -loptimizers -L../../neuralnet -lneuralnet -L../../c_npy -lc_npy -lomp -lm
