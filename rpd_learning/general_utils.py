#!/usr/bin/env python

import os
import shutil

_DTYPE_ORDERING = [
    'bool8',
    'ubyte',   'uint8',   'uint16',  'uint32',  'uint64',
    'byte',    'int8',    'int16',   'int32',   'int64',
    'float16', 'float32', 'float64', 'float96', 'float128'
]


def rm_rf(dir, confirmation_prompt=None):
    """Remove a directory and all of its contents.
    Warning: this is potentially a very dangerous operation.
    """
    if type(confirmation_prompt) == str:
        try:
            confirmation = raw_input('%s ' % confirmation_prompt)
        except NameError:
            confirmation = input('%s ' % confirmation_prompt)
    else:
        confirmation = True
    if (isinstance(confirmation, bool) and confirmation) \
            or (isinstance(confirmation, str) and confirmation.lower() == 'true'):
        for filename in os.listdir(dir):
            filepath = os.path.join(dir, filename)
            try:
                if os.path.isfile(filepath):
                    os.unlink(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
            except Exception as e:
                print(e)
        print('Successfully removed everything inside of `%s`.' % dir)
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
