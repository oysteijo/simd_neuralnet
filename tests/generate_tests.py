import neuralnet as nn
import metrics as refmetrics
import numpy as np

heading = """/* {} -- Automatically generated testfile. Modify '{}' to make changes */"""

includes = """
#include "test.h"
#include "{}.h"
#include "simd.h"
#include <stdio.h>
#include <string.h>
"""

main_start = """
int main(int argc, char *argv[] )
{
    if(argc == 1)
        fprintf(stderr, KBLU "Running '%%s'\\n" KNRM, argv[0] );
    int test_count = 0;
    int fail_count = 0;

    %s_func a;
    char *name;
"""

main_end ="""
    print_test_summary(test_count, fail_count );
    return 0;
}"""

activations = [
    "sigmoid",
    "softmax",
    "tanh",
    "relu",
    "hard_sigmoid",
    "softplus",
    "linear",
    "softsign",
    "exponential"]

losses = [
    "mean_squared_error" ,
    "mean_absolute_error" ,
    "mean_absolute_percentage_error" ,
    "binary_crossentropy" ,
    "categorical_crossentropy"]

metrics = [
    "mean_squared_error" ,
    "mean_absolute_error" ,
    "mean_absolute_percentage_error" ,
    "binary_crossentropy" ,
    "categorical_crossentropy" ,
    "binary_accuracy"]

def generate_activation_test( activation, testvec ):
    test_code = """
    /* Tests for '{0}' */
    a = get_activation_func("{0}");
    CHECK_NOT_NULL_MSG( a, "Check if activation '{0}' was found" );

    a_deriv = get_activation_derivative( a );
    CHECK_NOT_NULL_MSG( a_deriv, "Check if activation derivative for '{0}' was found" );

    name = (char*) get_activation_name(a);
    CHECK_CONDITION_MSG( strcmp( name, "{0}" ) == 0,
        "Checking name match of activation function '{0}'" );\n"""

    retstr = test_code.format( activation )

    py_activation = nn.get_activation_func( activation )
    for v in testvec:
        retstr += "    {\n"
        retstr += "        float SIMD_ALIGN(x[{0}]) = {{ {1} }};\n".format(len(v), ", ".join( [str( elem )+"f" for elem in v ] ))
        retstr += "        a( {}, x );\n".format(len(v))
    
        test_out = py_activation( v )
        for i, y in enumerate( test_out ):
            retstr += "        CHECK_FLOAT_EQUALS_MSG( {}f, x[{}], 1.0e-6, \"Comparing activation '{}' outputs\");\n".format(y, i, activation)
        retstr += "    }\n"
    return retstr

def generate_loss_test( loss, testvec ):
    test_code = """
    /* Tests for '{0}' */
    a = get_loss_func("{0}");
    CHECK_NOT_NULL_MSG( a, "Check if loss '{0}' was found" );

    name = (char*) get_loss_name(a);
    CHECK_CONDITION_MSG( strcmp( name, "{0}" ) == 0,
        "Checking name match of loss function '{0}'" );\n"""

    retstr = test_code.format( loss )

    py_loss = nn.get_loss_func( loss )
    for v in testvec:
        # Absolute percentage error gets problem then the real value is close to 0 as it divides by real. We hence add one.
        if "absolute_percentage" in loss:
            v = 1 + v
        retstr += "    {\n"
        retstr += "        float SIMD_ALIGN(y_pred[{0}]) = {{ {1} }};\n".format(len(v), ", ".join( [str( elem )+"f" for elem in v ] ))
        retstr += "        float SIMD_ALIGN(y_real[{0}]) = {{ {1} }};\n".format(len(v), ", ".join( [str( elem*0.95 )+"f" for elem in v ] ))
        retstr += "        float SIMD_ALIGN(loss[{0}]);\n".format(len(v))
        retstr += "        a( {}, y_pred, y_real, loss );\n".format(len(v))
    
        test_out = py_loss( v, v*0.95 )
        for i, y in enumerate( test_out ):
            retstr += "        CHECK_FLOAT_EQUALS_MSG( {}f, loss[{}], 1.0e-6, \"Comparing loss '{}' outputs\");\n".format(y, i, loss)
        retstr += "    }\n"
    return retstr

def generate_metrics_test( metric, testvec ):
    test_code = """
    /* Tests for '{0}' */
    a = get_metric_func("{0}");
    CHECK_NOT_NULL_MSG( a, "Check if metrics '{0}' was found" );

    name = (char*) get_metric_name(a);
    CHECK_CONDITION_MSG( strcmp( name, "{0}" ) == 0,
        "Checking name match of metrics function '{0}'" );\n"""

    retstr = test_code.format( metric )

    py_metrics = refmetrics.get_metric_func( metric )
    for v in testvec:
        # Mean absolute percentage error as this is scaling by 0.95. It will therefore always git 5% error.
        if "absolute_percentage" in metric:
            break
        if "accuracy" in metric:
            break
        retstr += "    {\n"
        retstr += "        float SIMD_ALIGN(y_pred[{0}]) = {{ {1} }};\n".format(len(v), ", ".join( [str( elem )+"f" for elem in v ] ))
        retstr += "        float SIMD_ALIGN(y_real[{0}]) = {{ {1} }};\n".format(len(v), ", ".join( [str( elem*0.95 )+"f" for elem in v ] ))
        retstr += "        float val = a( {}, y_pred, y_real );\n".format(len(v))
    
        test_out = py_metrics( v, v*0.95 )
        retstr += "        CHECK_FLOAT_EQUALS_MSG( {}f, val, 1.0e-6, \"Comparing metrics '{}' outputs\");\n".format(test_out, metric)
        retstr += "    }\n"

    if "absolute_percentage" in metric:
        y_real = np.arange(10, 20, dtype=np.float32)             # Number that are off zero
        y_pred = y_real + np.random.random(y_real.shape) - 0.5   # some noise added 

        retstr += "    {\n"
        retstr += "        float SIMD_ALIGN(y_pred[{0}]) = {{ {1} }};\n".format(len(y_pred), ", ".join( [str( elem )+"f" for elem in y_pred ] ))
        retstr += "        float SIMD_ALIGN(y_real[{0}]) = {{ {1} }};\n".format(len(y_real), ", ".join( [str( elem )+"f" for elem in y_real ] ))
        retstr += "        float val = a( {}, y_pred, y_real );\n".format(len(y_pred))

        test_out = py_metrics( y_pred, y_real )
        retstr += "        CHECK_FLOAT_EQUALS_MSG( {}f, val, 1.0e-4, \"Comparing metrics '{}' outputs\");\n".format(test_out, metric)
        #retstr += "        fprintf(stderr, \"{}f (python) ?= %g (simd_neuralnet)\\n\", val);\n".format(test_out)
        retstr += "    }\n"
    
    if "accuracy" in metric:
        for y_pred in testvec:
            y_real = np.random.choice([0,], size=y_pred.shape ).astype(np.float32)
    
            retstr += "    {\n"
            retstr += "        float SIMD_ALIGN(y_pred[{0}]) = {{ {1} }};\n".format(len(y_pred), ", ".join( [str( elem )+"f" for elem in y_pred ] ))
            retstr += "        float SIMD_ALIGN(y_real[{0}]) = {{ {1} }};\n".format(len(y_real), ", ".join( [str( elem )+"f" for elem in y_real ] ))
            retstr += "        float val = a( {}, y_pred, y_real );\n".format(len(y_pred))
    
            test_out = py_metrics( y_pred, y_real )
            retstr += "        CHECK_FLOAT_EQUALS_MSG( {}f, val, 1.0e-4, \"Comparing metrics '{}' outputs\");\n".format(test_out, metric)
            #retstr += "        fprintf(stderr, \"{}f (python) ?= %g (simd_neuralnet)\\n\", val);\n".format(test_out)
            retstr += "    }\n"
    
    return retstr

if __name__ == '__main__':
    import sys
    import os
    generate_tests = {"activation": "activations", "loss": "losses", "metrics": "metrics"}
    for obj in generate_tests.keys():
        with open( "test_{}.c".format(obj), "w+" ) as f:
            f.write( heading.format(os.path.basename(f.name), __file__ )  )
            f.write( includes.format( obj ))
            if obj == "metrics":
                f.write( main_start % obj[:-1] )
            else:
                f.write( main_start % obj )

            if obj == "activation":
                f.write( "    activation_derivative a_deriv;\n" )

            for a in getattr(sys.modules[__name__], generate_tests[obj]):
                vec = [
                    np.random.random( (4,)).astype(np.float32 ),
                    np.random.random( (9,)).astype(np.float32 ),
                    np.random.random( (17,)).astype(np.float32 ),
                ]
                genfunc = getattr(sys.modules[__name__], "generate_%s_test" % obj )
                f.write(genfunc( a, vec ))
                # In addition we add a test to check the numerical stability of softmax
            if obj == "activation":
                f.write(generate_activation_test( "softmax", [np.array([1000,2000,3000], dtype=np.float32)] ))  # This gives overflow on naiive implementations  
                f.write(generate_activation_test( "sigmoid", [np.array([-1000,1000], dtype=np.float32)] ))  # This gives NaN on naiive implementations  
            f.write(main_end)

