#!/usr/bin/env python

"""
general_utilities.py

A collection of general utilities.
"""

import shutil
import numpy as np

_DTYPE_ORDERING = [
    'bool8',
    'ubyte',   'uint8',   'uint16',  'uint32',  'uint64',
    'byte',    'int8',    'int16',   'int32',   'int64',
    'float16', 'float32', 'float64', 'float96', 'float128'
]


def merge_dicts(*args):
    z = args[0].copy()  # start with the first dictionary's keys and values
    for y in args[1:]:
        z.update(y)  # modifies z with y's keys and values & returns None
    return z


def confirm(prompt):
    """Return True if the user enters 'true' and False otherwise."""
    try:
        confirmation = raw_input('%s ' % prompt)
    except NameError:
        confirmation = input('%s ' % prompt)
    if isinstance(confirmation, bool):
        return confirmation
    return isinstance(confirmation, str) and confirmation.lower() == 'true'


def rm_rf(dir, confirmation_prompt=None):
    """Remove a directory and all of its contents.
    Warning: this is potentially a very dangerous operation.
    """
    confirmation = confirm(confirmation_prompt) if type(confirmation_prompt) == str else True
    if confirmation:
        shutil.rmtree(dir)
        print('Successfully removed `%s` and all of its contents.' % dir)
    else:
        print('Operation `rm -rf %s` aborted.' % dir)


def dominant_dtype(dtypes):
    """Given a list of strings representing data types, returns the most dominant data type from the list.
    For example, 'float32' would dominate 'int32'.

    This function does not check data type compatibility.
    If a 'uint64' and a 'float16' are passed in, it will return 'float16' with no qualms whatsoever.
    """
    dominant, max_priority = None, -1
    for dtype in dtypes:
        try:
            priority = _DTYPE_ORDERING.index(dtype)
        except ValueError:
            error_msg = 'ERROR: dtype %s not supported yet.\n' % dtype
            error_msg += 'Go into `general_utils.py` and add %s to _DTYPE_ORDERING.' % dtype
            print(error_msg)
            raise
        if priority > max_priority:
            dominant, max_priority = dtype, priority
    return dominant


def eval_keys(d, _globals=None, _locals=None):
    """Replaces any string keys of a dictionary with their evaluated forms."""
    if _globals is None:
        _globals = globals()
    if _locals is None:
        _locals = _globals
    keys = d.keys()
    for key in keys:
        if type(key) == str:
            value = d[key]
            del d[key]
            d[eval(key, _globals, _locals)] = value
    return d


def sum_functions(*fns):
    """Return a new function F(x1, ...) that, when called, returns the result of summing
    F1(x1, ...), F2(x1, ...), ..., FN(x1, ...) - where F1, ..., FN are the N functions given
    by FNS.

    In other words, F(x1, ...) = F1(x1, ...) + F2(x1, ...) + ... + FN(x1, ...)

    Each function is assumed to have the same signature.
    """
    if len(fns) == 1 and type(fns[0]) in (tuple, list):
        fns = fns[0]  # can call `sum_functions` as `sum_functions(f1, f2, ...)` or `sum_functions([f1, f2, ...])`
    return lambda *args: sum([fn(*args) for fn in fns])


def sum_functions_with_weights(fns, weights):
    """Like `sum_functions` where the return value of each function from FNS
    is weighted by the same-indexed term from WEIGHTS.

    In other words, if FNS = [f1, f2, ..., fn] and WEIGHTS = [w1, w2, ..., wn],
    then the returned function F(...) = w1 * f1(...) + w2 * f2(...) + ... + wn * fn(...).

    It is assumed that `len(fns)` == `len(weights)`.
    """
    assert len(fns) == len(weights), 'must have exactly one weight for every function'
    return lambda *args: sum([w * fn(*args) for fn, w in zip(fns, weights)])


def next_greater(iterable, x):
    """Returns from ITERABLE the closest element to X that is also greater than X.
    If no elements in ITERABLE are greater than X, returns None.
    """
    iterable_np = np.array(iterable)
    all_greater = iterable_np[iterable_np > x]
    return all_greater.min() if len(all_greater) > 0 else None
