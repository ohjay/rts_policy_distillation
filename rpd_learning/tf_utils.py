#!/usr/bin/env python

"""
tf_utils.py

A collection of TensorFlow utilities.
"""

import tensorflow as tf


def minimize_and_clip(optimizer, objective, var_list, clip_val=10, is_training=True):
    """Minimize OBJECTIVE using OPTIMIZER w.r.t. variables in VAR_LIST,
    while ensuring that the norm of the gradients for each variable is clipped to CLIP_VAL.
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) if is_training else []
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(gradients)
    return train_op


def initialize_interdependent_variables(session, vars_list, feed_dict):
    """Initialize a list of variables one at a time, which is useful if
    initialization of some variables depends on initialization of the others.
    """
    vars_left = vars_list
    while len(vars_left) > 0:
        new_vars_left = []
        for v in vars_left:
            try:
                session.run(tf.variables_initializer([v]), feed_dict)
            except tf.errors.FailedPreconditionError:
                new_vars_left.append(v)
        if len(new_vars_left) >= len(vars_left):
            # This can happen if the variables all depend on each other, or more likely if there's
            # another variable outside of the list that still needs to be initialized. This could be
            # detected here, but life's finite.
            raise Exception('Cycle in variable dependencies, or external precondition unsatisfied.')
        else:
            vars_left = new_vars_left
