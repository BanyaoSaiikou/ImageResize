#! /usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer

class MultiSGD(Optimizer):
    def __init__(self, learning_rates, name="MultiSGD", **kwargs):
        super(MultiSGD, self).__init__(name, **kwargs)
        self.learning_rates = learning_rates

    def apply_gradients(self, grads_and_vars, name=None):
        updates = []
        for (grad, param), learning_rate in zip(grads_and_vars, self.learning_rates):
            if grad is not None:
                update = -learning_rate * grad
                updates.append(tf.assign_add(param, update))
        return tf.group(*updates, name=name)

