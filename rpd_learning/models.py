#!/usr/bin/env python

"""
models.py

Utilities for constructing a model.
"""

import os
import copy
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, convolution2d, flatten, batch_norm

import ray


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


class A3CPolicy(object):
    """Specialized policy for the A3C method."""

    def __init__(self, arch, scope='local', summarize=True):
        # Read arch config (TODO: set up actual graph in architecture)
        input_names = sorted(arch['inputs'].keys())
        assert len(input_names) == 1, 'only one observation allowed right now (TODO)'
        input_shapes = [arch['inputs'][input_name]['shape'] for input_name in input_names]
        output_names = sorted(arch['outputs'].keys(), key=lambda x: arch['outputs'][x].get('order', float('inf')))
        output_shapes = [arch['outputs'][output_name]['shape'] for output_name in output_names]
        input_shape = input_shapes[0]  # TODO: hacky... we're assuming one input
        output_shape = output_shapes[0]  # TODO: hacky... we're assuming one output

        self.local_steps = 0
        self.summarize = summarize
        worker_device = "/job:localhost/replica:0/task:0/cpu:0"
        self.g = tf.Graph()
        with self.g.as_default(), tf.device(worker_device):
            with tf.variable_scope(scope):
                self._setup_graph(input_shape, output_shape)
                assert all([hasattr(self, attr) for attr in ["vf", "logits", "x", "var_list"]])
            self.setup_loss(output_shape)
            self.setup_gradients()
            self.initialize()

    def _setup_graph(self, input_shape, output_shape):
        self.x = tf.placeholder(tf.float32, [None] + list(input_shape))

        out = self.x
        with tf.variable_scope("convnet"):
            out = convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = flatten(out)
        out = fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu)
        self.logits = fully_connected(out, num_outputs=output_shape[0], activation_fn=None)
        self.action_probs = tf.nn.softmax(self.logits)
        self.vf = fully_connected(out, num_outputs=1, activation_fn=None)

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          tf.get_variable_scope().name)
        self.global_step = tf.get_variable("global_step", [], tf.int32,
                                           initializer=tf.constant_initializer(0, dtype=tf.int32), trainable=False)

    def setup_loss(self, output_shape):
        print('Setting up loss.')
        self.ac = tf.placeholder(tf.float32, [None], name="ac")
        self.adv = tf.placeholder(tf.float32, [None], name="adv")
        self.r = tf.placeholder(tf.float32, [None], name="r")

        log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=tf.cast(self.ac, tf.int32))

        # The "policy gradients" loss: its derivative is precisely the policy
        # gradient. Notice that self.ac is a placeholder that is provided
        # externally. adv will contain the advantages, as calculated in `process_rollout`.
        self.pi_loss = - tf.reduce_sum(log_prob * self.adv)

        delta = self.vf - self.r
        self.vf_loss = 0.5 * tf.reduce_sum(tf.square(delta))

        # Compute entropy
        a0 = self.logits - tf.reduce_max(self.logits, reduction_indices=[1], keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, reduction_indices=[1], keep_dims=True)
        p0 = ea0 / z0
        self.entropy = tf.reduce_sum(tf.reduce_sum(p0 * (tf.log(z0) - a0), reduction_indices=[1]))

        self.loss = self.pi_loss + 0.5 * self.vf_loss - self.entropy * 0.01

    def setup_gradients(self):
        grads = tf.gradients(self.loss, self.var_list)
        self.grads, _ = tf.clip_by_global_norm(grads, 40.0)
        grads_and_vars = list(zip(self.grads, self.var_list))
        opt = tf.train.AdamOptimizer(1e-4)
        self._apply_gradients = opt.apply_gradients(grads_and_vars)

    def initialize(self):
        if self.summarize:
            bs = tf.to_float(tf.shape(self.x)[0])
            tf.summary.scalar("model/policy_loss", self.pi_loss / bs)
            tf.summary.scalar("model/value_loss", self.vf_loss / bs)
            tf.summary.scalar("model/entropy", self.entropy / bs)
            tf.summary.scalar("model/grad_gnorm", tf.global_norm(self.grads))
            tf.summary.scalar("model/var_gnorm", tf.global_norm(self.var_list))
            self.summary_op = tf.summary.merge_all()

        self.sess = tf.Session(graph=self.g, config=tf.ConfigProto(
            intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
        self.variables = ray.experimental.TensorFlowVariables(self.loss, self.sess)
        self.sess.run(tf.global_variables_initializer())

    def apply_gradients(self, grads):
        feed_dict = {self.grads[i]: grads[i]
                     for i in range(len(grads))}
        self.sess.run(self._apply_gradients, feed_dict=feed_dict)

    def get_weights(self):
        weights = self.variables.get_weights()
        return weights

    def set_weights(self, weights):
        self.variables.set_weights(weights)

    def compute_gradients(self, batch):
        info = {}
        feed_dict = {
            self.x: batch.si,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
        }
        self.grads = [g for g in self.grads if g is not None]
        self.local_steps += 1
        if self.summarize:
            grad, summ = self.sess.run([self.grads, self.summary_op],
                                       feed_dict=feed_dict)
            info['summary'] = summ
        else:
            grad = self.sess.run(self.grads, feed_dict=feed_dict)
        return grad, info

    def compute_action(self, ob):
        action_probs, vf = self.sess.run([self.action_probs, self.vf], {self.x: [ob]})
        action = np.argmax(action_probs[0].flatten())
        return action, {'value': vf[0]}

    def value(self, ob):
        vf = self.sess.run(self.vf, {self.x: [ob]})
        return vf[0]
