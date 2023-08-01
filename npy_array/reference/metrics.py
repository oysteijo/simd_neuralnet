import numpy as np
import sys

def get_metric_func(s):
    return getattr(sys.modules[__name__], s)

mean_squared_error             = lambda y_pred, y_real: np.mean(np.square(y_pred - y_real), axis=-1)
mean_absolute_error            = lambda y_pred, y_real: np.mean(np.abs(y_pred - y_real), axis=-1)
mean_absolute_percentage_error = lambda y_pred, y_real: 100 * np.mean( np.abs(( y_pred - y_real ) / np.clip( np.abs(y_real), 1.0e-7, None)), axis=-1)
binary_crossentropy            = lambda y_pred, y_real: -np.mean(y_real*np.log(y_pred)+(1.0-y_real)*(np.log(1.0-y_pred)), axis=-1)
categorical_crossentropy       = lambda y_pred, y_real: -np.mean(y_real*np.log(y_pred), axis=-1)
binary_accuracy                = lambda y_pred, y_real: np.mean(np.clip( y_pred, 0.0, 1.0 ).round() == np.clip( y_real, 0.0, 1.0 ).round(), axis=-1)
