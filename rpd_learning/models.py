#!/usr/bin/env python

"""
models.py

Utilities for constructing a model.
"""

import os
import copy
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, convolution2d, flatten, batch_norm

class Model(object):
    curr_index = -1

    def __init__(self, arch, inputs=None, scope=None, reuse=False):
        self.inputs = inputs if type(inputs) == dict else {}
        self.outputs, self.labels = {}, {}
        self.losses, self.optimize = {}, {}

        # Construct the actual graph
        if scope is None:
            Model.curr_index += 1
        self.scope = scope or 'Model_%d' % Model.curr_index
        with tf.variable_scope(self.scope, reuse=reuse):
            self.load_arch(arch)

    def load_arch(self, arch):
        """Build the computational graph according to the given architecture."""
        with tf.variable_scope('inputs'):
            for input_name, input_info in arch['inputs'].items():
                if input_name not in self.inputs:
                    self.inputs[input_name] = self.load_placeholder(input_info)

        # Set up the architecture
        layer_outputs = self.load_layers(arch['layers'])

        # Connect the outputs
        inputs = layer_outputs[-1] if layer_outputs else self.inputs.values()
        inputs = tf.concat(inputs, axis=1) if len(inputs) > 1 else inputs[-1]
        with tf.variable_scope('outputs'):
            for output_name, output_info in arch['outputs'].items():
                self.outputs[output_name] = self.load_output(output_name, output_info, inputs)

        print('[+] Finished building the graph.')

    @staticmethod
    def load_placeholder(placeholder_info):
        dtype = placeholder_info['dtype']
        shape = [None] + list(placeholder_info['shape'])
        return tf.placeholder(getattr(tf, dtype), shape=shape)

    def load_layers(self, layers):
        outputs = []
        for i, layer in enumerate(layers):
            outputs.append([])
            with tf.variable_scope('layer%d' % i):
                if type(layer) == list:
                    layer_output = self.load_layers(layer)
                    layer_output = [item for sublist in layer_output for item in sublist]  # flatten
                    outputs[i].extend(layer_output)
                else:
                    if i > 0:
                        inputs = copy.deepcopy(layer.get('inputs', [])) + outputs[i - 1]  # use output of previous layer as default
                    else:
                        inputs = copy.deepcopy(layer['inputs'])  # inputs required for first layer
                    for j in range(len(inputs)):
                        if type(inputs[j]) == str:
                            inputs[j] = self.inputs[inputs[j]]
                    num_outputs = layer['num_outputs']
                    try:
                        activation_fn = getattr(tf.nn, layer['activation'])
                    except AttributeError:
                        activation_fn = None
                    if 'biases_init' in layer:
                        biases_initializer = tf.constant_initializer(layer['biases_init'])
                    else:
                        biases_initializer = tf.zeros_initializer()  # default

                    layer_type = layer['type']
                    inputs = tf.concat(inputs, axis=1) if len(inputs) > 1 else inputs[-1]
                    if layer_type == 'fc':
                        _output = fully_connected(inputs, num_outputs=num_outputs, activation_fn=None,
                                                  biases_initializer=biases_initializer)
                    elif layer_type == 'conv2d':
                        _output = convolution2d(inputs, num_outputs=num_outputs, kernel_size=layer['kernel_size'],
                                                stride=layer['stride'], activation_fn=None,
                                                biases_initializer=biases_initializer)
                    else:
                        raise NotImplementedError
                    if layer.get('batch_norm', False):
                        _output = batch_norm(_output, is_training=True)  # TODO: set `is_training` via flag or something
                    if activation_fn is not None:
                        _output = activation_fn(_output)
                    outputs[i].append(_output)
                    if layer_type == 'conv2d' and i < len(layers) and layers[i + 1]['type'] == 'fc':
                        # If convolution layer into fully connected layer, we need to flatten this output
                        outputs[i][-1] = flatten(outputs[i][-1])
                    elif layer_type == 'fc' and i == len(layers) - 1:
                        # If FC and last layer, flatten (TODO check this with Atari Pong - does this break things?)
                        outputs[i][-1] = flatten(outputs[i][-1])
        return outputs

    def load_output(self, output_name, output_info, inputs):
        shape = output_info['shape']
        assert len(shape) == 0 or len(shape) == 1
        num_outputs = shape[0] if shape else 1
        try:
            activation_fn = getattr(tf.nn, output_info.get('activation', 'relu'))
        except AttributeError:
            activation_fn = None
        if 'biases_init' in output_info:
            biases_initializer = tf.constant_initializer(output_info['biases_init'])
        else:
            biases_initializer = tf.zeros_initializer()  # default
        outputs = fully_connected(inputs, num_outputs, activation_fn=activation_fn, biases_initializer=biases_initializer)

        # Loss
        loss = output_info.get('loss', None)
        if loss is not None and loss != 'tbd':
            for loss_type, weight in loss['terms'].items():
                self.losses[output_name] = tf.constant(0.0)
                if output_info.get('labeled', False):
                    self.labels[output_name] = self.load_placeholder(output_info)
                    self.losses[output_name] += weight * self.compute_loss(loss_type, self.labels[output_name], outputs)
                else:
                    raise NotImplementedError
            self.losses[output_name] += loss.get('reg', 0.0) * sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            tf.summary.scalar('%s/loss' % output_name, self.losses[output_name])

        # Optimization
        opt = output_info.get('opt', None)
        if opt is not None and opt != 'tbd':
            lr = opt.get('lr', 1e-3)
            _optimizer = tf.train.AdamOptimizer(lr)
            self.optimize[output_name] = _optimizer.minimize(self.losses[output_name])

        return outputs

    @staticmethod
    def compute_loss(loss_type, labels, outputs):
        if loss_type == 'l1':
            return tf.reduce_mean(tf.abs(outputs - labels))
        elif loss_type == 'l2':
            return tf.reduce_mean(tf.squared_difference(outputs, labels))
        raise NotImplementedError

    def save(self, sess, iteration, outfolder='out', write_meta_graph=True):
        """Save the model's variables (as they exist in the current session) to a checkpoint file in OUTFOLDER."""
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        base_filepath = os.path.join(outfolder, 'var')
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))
        saver.save(sess, base_filepath, global_step=iteration, write_meta_graph=write_meta_graph)
        print('[+] Saved current parameters to %s-%d.index.' % (base_filepath, iteration))

    def restore(self, sess, iteration, outfolder='out'):
        """Restore the model's variables from a checkpoint file in OUTFOLDER."""
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))
        saver.restore(sess, os.path.join(outfolder, 'var-%d' % iteration))
        print('[+] Model restored to iteration %d (outfolder=%s).' % (iteration, outfolder))
