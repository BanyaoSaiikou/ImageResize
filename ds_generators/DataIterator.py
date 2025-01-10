#! /usr/bin/env python
# coding: utf-8

import os
import h5py
import numpy as np
from tensorflow.keras.utils import Sequence

class DataIterator(Sequence):
    def __init__(self, data_file, batch_size=32, shuffle=True):
        self.data_file = data_file
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file = h5py.File(data_file, 'r')
        self.num_samples = self.file['images'].shape[0]  # Assuming you have 'images' and 'labels' datasets
        self.indexes = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_indexes = self.indexes[start:end]

        batch_images, batch_labels = self.load_data(batch_indexes)

        return batch_images, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def load_data(self, batch_indexes):
        batch_images = self.file['images'][batch_indexes]
        batch_labels = self.file['labels'][batch_indexes]

        return batch_images, batch_labels


