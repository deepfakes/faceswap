#!/usr/bin/env python3
""" Utils imported from Keras as their location changes between Tensorflow Keras and standard
Keras. Also ensures testing consistency """
import inspect

import numpy as np


def generate_test_data(num_train=1000, num_test=500, input_shape=(10,),
                       output_shape=(2,),
                       classification=True, num_classes=2):
    """Generates test data to train a model on. classification=True overrides output_shape (i.e.
    output_shape is set to (1,)) and the output consists in integers in [0, num_classes-1].

    Otherwise: float output with shape output_shape.
    """
    samples = num_train + num_test
    if classification:
        var_y = np.random.randint(0, num_classes, size=(samples,))
        var_x = np.zeros((samples,) + input_shape, dtype=np.float32)
        for i in range(samples):
            var_x[i] = np.random.normal(loc=var_y[i], scale=0.7, size=input_shape)
    else:
        y_loc = np.random.random((samples,))
        var_x = np.zeros((samples,) + input_shape, dtype=np.float32)
        var_y = np.zeros((samples,) + output_shape, dtype=np.float32)
        for i in range(samples):
            var_x[i] = np.random.normal(loc=y_loc[i], scale=0.7, size=input_shape)
            var_y[i] = np.random.normal(loc=y_loc[i], scale=0.7, size=output_shape)

    return (var_x[:num_train], var_y[:num_train]), (var_x[num_train:], var_y[num_train:])


def to_categorical(var_y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.

    Parameters
    ----------
    var_y: int
        Class vector to be converted into a matrix (integers from 0 to num_classes).
    num_classes: int
        Total number of classes.
    dtype: str
        The data type expected by the input, as a string (`float32`, `float64`, `int32`...)

    Returns
    -------
    tensor
        A binary matrix representation of the input. The classes axis is placed last.

    Example
    -------
    >>> # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    >>> labels
    >>> array([0, 2, 1, 2, 0])
    >>> # `to_categorical` converts this into a matrix with as many columns as there are classes.
    >>> # The number of rows stays the same.
    >>> to_categorical(labels)
    >>> array([[ 1.,  0.,  0.],
    >>>        [ 0.,  0.,  1.],
    >>>        [ 0.,  1.,  0.],
    >>>        [ 0.,  0.,  1.],
    >>>        [ 1.,  0.,  0.]], dtype=float32)
    """
    var_y = np.array(var_y, dtype='int')
    input_shape = var_y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    var_y = var_y.ravel()
    if not num_classes:
        num_classes = np.max(var_y) + 1
    var_n = var_y.shape[0]
    categorical = np.zeros((var_n, num_classes), dtype=dtype)
    categorical[np.arange(var_n), var_y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def has_arg(func, name, accept_all=False):
    """Checks if a callable accepts a given keyword argument.

    For Python 2, checks if there is an argument with the given name.
    For Python 3, checks if there is an argument with the given name, and also whether this
    argument can be called with a keyword (i.e. if it is not a positional-only argument).

    Parameters
    ----------
    func: object
        Callable to inspect.
    name: str
        Check if `func` can be called with `name` as a keyword argument.
    accept_all: bool, optional
        What to return if there is no parameter called `name` but the function accepts a
        `**kwargs` argument. Default: ``False``

    Returns
    -------
    bool
        Whether `func` accepts a `name` keyword argument.
    """
    signature = inspect.signature(func)
    parameter = signature.parameters.get(name)
    if parameter is None:
        if accept_all:
            for param in signature.parameters.values():
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    return True
        return False
    return (parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                               inspect.Parameter.KEYWORD_ONLY))
