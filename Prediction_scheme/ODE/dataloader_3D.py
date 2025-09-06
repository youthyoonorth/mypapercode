import os
from os import listdir
from os.path import isfile, join
import scipy.io as sio
import numpy as np
import torch


class Dataloader_3D():
    def __init__(self, path='', batch_size=32, device='cpu'):
        self.batch_size = batch_size
        self.device = device
        # gather all .mat files in the provided directory
        self.files = [
            join(path, f)
            for f in listdir(path)
            if isfile(join(path, f)) and f.endswith('.mat')
        ]
        self.reset()

    def reset(self):
        self.done = False
        self.unvisited_files = [f for f in self.files]

        # determine number of beams from the first file
        self.num_beams = 0
        if self.unvisited_files:
            sample = sio.loadmat(self.unvisited_files[0])
            self.num_beams = sample['beam_power'].shape[2]

        # batch_size * 2 * length * num of beam
        self.buffer = np.zeros((0, 2, 101, self.num_beams))

        # batch_size * length
        self.buffer_label = np.zeros((0, 101))

        # batch_size * length * num of beam
        self.buffer_beam_power = np.zeros((0, 101, self.num_beams))

    def load(self, file):
        data = sio.loadmat(file)

        channels = data['MM_data']  # beam training received signal
        labels = data['beam_label'] - 1  # optimal beam index label
        beam_power = data['beam_power']  # beam amplitude

        # ensure the beam dimension matches the expected value
        num_beams = beam_power.shape[2]
        assert (
            num_beams == self.num_beams
        ), f"Inconsistent beam dimension in {file}: expected {self.num_beams}, got {num_beams}"

        return channels, labels, beam_power

    def next_batch(self):
        done = False
        count = True

        # sequentially load data
        while self.buffer.shape[0] < self.batch_size:
            if len(self.unvisited_files) == 0:
                done = True
                count = False
                break
            channels, labels, beam_power = self.load(
                self.unvisited_files.pop(0))

            # load data into buffers
            self.buffer = np.concatenate((self.buffer, channels), axis=0)
            self.buffer_label = np.concatenate((self.buffer_label, labels), axis=0)
            self.buffer_beam_power = np.concatenate((self.buffer_beam_power, beam_power), axis=0)

        out_size = min(self.batch_size, self.buffer.shape[0])
        # get data from buffers
        batch_channels = self.buffer[0:out_size, :, :, :]
        batch_labels = self.buffer_label[0:out_size, :]
        batch_beam_power = self.buffer_beam_power[0:out_size, :, :]

        self.buffer = np.delete(self.buffer, np.s_[0 : out_size], 0)
        self.buffer_label = np.delete(self.buffer_label, np.s_[0 : out_size], 0)
        self.buffer_beam_power = np.delete(self.buffer_beam_power, np.s_[0:out_size], 0)

        # format transformation for reducing overhead
        batch_channels = np.float32(batch_channels)
        batch_labels = batch_labels.astype(int)
        batch_beam_power = np.float32(batch_beam_power)

        return torch.from_numpy(batch_channels).to(self.device), torch.from_numpy(batch_labels).to(
            self.device), torch.from_numpy(batch_beam_power).to(
            self.device), done, count