import torch
import numpy as np
import pandas as pd
import math
from tensorflow.keras.utils import Sequence


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, df):
        self.file_name = df["FILENAME"].values
        self.gender = df["GENDER"].values
        self.age = df["AGE"].values
        self.file_path = file_path

    def __getitem__(self, index):
        file_name = self.file_name[index]
        full_array = np.load(self.file_path + file_name + ".npy")
        _12lead = [
            "I",
            "II",
            "III",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
            "aVR",
            "aVL",
            "aVF",
        ]
        array_by_name = dict()
        for name, p_array in zip(_12lead, np.split(full_array, 12)):
            array_by_name[name] = p_array
        _12lead_re = [
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]
        full_array = []
        for _array in _12lead_re:
            full_array += array_by_name[_array].tolist()
        full_array = np.array(full_array)
        label = self.age[index]
        return full_array, array_by_name, label

    def __len__(self):
        return len(self.file_name)


class TF_Dataloader(Sequence):
    def __init__(self, file_path, df, batch_size, shuffle=False):
        self.file_name = df["FILENAME"].values
        self.gender = df["GENDER"].values
        self.age = df["AGE"].values
        self.file_path = file_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.file_name) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]

        file_name = [self.file_name[i] for i in indices]
        arrays = []
        for name in file_name:
            orig_array = np.load(self.file_path + name + ".npy")
            _12lead = [
                "I",
                "II",
                "III",
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
                "aVR",
                "aVL",
                "aVF",
            ]
            array_by_name = dict()
            for name, p_array in zip(_12lead, np.split(orig_array, 12)):
                array_by_name[name] = p_array
            _12lead_re = [
                "I",
                "II",
                "III",
                "aVR",
                "aVL",
                "aVF",
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
            ]
            full_array = []
            for _array in _12lead_re:
                full_array += array_by_name[_array].tolist()
            arrays.append(full_array)
        arrays = np.swapaxes(np.array(arrays).reshape(-1, 12, 5000), 1, 2)
        label = np.array([self.age[i] for i in indices])

        return arrays, label

    # epoch이 끝날때마다 실행
    def on_epoch_end(self):
        self.indices = np.arange(len(self.file_name))
        if self.shuffle == True:
            np.random.shuffle(self.indices)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, df):
        self.file_name = df["FILENAME"].values
        self.gender = df["GENDER"].values
        self.file_path = file_path

    def __getitem__(self, index):
        file_name = self.file_name[index]
        full_array = np.load(self.file_path + file_name + ".npy")
        _12lead = [
            "I",
            "II",
            "III",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
            "aVR",
            "aVL",
            "aVF",
        ]
        array_by_name = dict()
        for name, p_array in zip(_12lead, np.split(full_array, 12)):
            array_by_name[name] = p_array
        _12lead_re = [
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]
        full_array = []
        for _array in _12lead_re:
            full_array += array_by_name[_array].tolist()
        full_array = np.array(full_array)
        return full_array, array_by_name

    def __len__(self):
        return len(self.file_name)


class CustomDataset_Norm(torch.utils.data.Dataset):
    def __init__(self, file_path, df):
        self.file_name = df["FILENAME"].values
        self.gender = df["GENDER"].values
        self.age = df["AGE"].values
        self.file_path = file_path

    def __getitem__(self, index):
        file_name = self.file_name[index]
        full_array = np.load(self.file_path + file_name + ".npy")
        full_array -= np.mean(full_array)
        full_array /= np.std(full_array) + 0.001
        _12lead = [
            "I",
            "II",
            "III",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
            "aVR",
            "aVL",
            "aVF",
        ]
        array_by_name = dict()
        for name, p_array in zip(_12lead, np.split(full_array, 12)):
            array_by_name[name] = p_array
        _12lead_re = [
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]
        full_array = []
        for _array in _12lead_re:
            full_array += array_by_name[_array].tolist()
        full_array = np.array(full_array)
        label = self.age[index]
        return full_array, array_by_name, label

    def __len__(self):
        return len(self.file_name)


class TF_Dataloader_Norm(Sequence):
    def __init__(self, file_path, df, batch_size, shuffle=False):
        self.file_name = df["FILENAME"].values
        self.gender = df["GENDER"].values
        self.age = df["AGE"].values
        self.file_path = file_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.file_name) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]

        file_name = [self.file_name[i] for i in indices]
        arrays = []
        for name in file_name:
            orig_array = np.load(self.file_path + name + ".npy")
            orig_array -= np.mean(orig_array)
            orig_array /= np.std(orig_array) + 0.001
            _12lead = [
                "I",
                "II",
                "III",
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
                "aVR",
                "aVL",
                "aVF",
            ]
            array_by_name = dict()
            for name, p_array in zip(_12lead, np.split(orig_array, 12)):
                array_by_name[name] = p_array
            _12lead_re = [
                "I",
                "II",
                "III",
                "aVR",
                "aVL",
                "aVF",
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
            ]
            full_array = []
            for _array in _12lead_re:
                full_array += array_by_name[_array].tolist()
            arrays.append(full_array)
        arrays = np.swapaxes(np.array(arrays).reshape(-1, 12, 5000), 1, 2)
        label = np.array([self.age[i] for i in indices])

        return arrays, label

    # epoch이 끝날때마다 실행
    def on_epoch_end(self):
        self.indices = np.arange(len(self.file_name))
        if self.shuffle == True:
            np.random.shuffle(self.indices)


class ad_TF_Dataloader_Norm(Sequence):
    def __init__(self, file_path, df, batch_size, shuffle=False):
        self.file_name = df["FILENAME"].values
        self.gender = df["GENDER"].values
        self.age = df["AGE"].values
        self.file_path = file_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.file_name) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]

        file_name = [self.file_name[i] for i in indices]
        arrays = []
        for name in file_name:
            orig_array = np.load(self.file_path + name + ".npy")
            orig_array -= np.mean(orig_array)
            orig_array /= np.std(orig_array) + 0.001
            arrays.append(orig_array)
        arrays = np.swapaxes(np.array(arrays).reshape(-1, 12, 5000), 1, 2)
        label = np.array([self.age[i] for i in indices])

        return arrays, label

    # epoch이 끝날때마다 실행
    def on_epoch_end(self):
        self.indices = np.arange(len(self.file_name))
        if self.shuffle == True:
            np.random.shuffle(self.indices)


class TestDataset_Norm(torch.utils.data.Dataset):
    def __init__(self, file_path, df):
        self.file_name = df["FILENAME"].values
        self.gender = df["GENDER"].values
        self.file_path = file_path

    def __getitem__(self, index):
        file_name = self.file_name[index]
        full_array = np.load(self.file_path + file_name + ".npy")
        full_array -= np.mean(full_array)
        full_array /= np.std(full_array) + 0.001
        _12lead = [
            "I",
            "II",
            "III",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
            "aVR",
            "aVL",
            "aVF",
        ]
        array_by_name = dict()
        for name, p_array in zip(_12lead, np.split(full_array, 12)):
            array_by_name[name] = p_array
        _12lead_re = [
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]
        full_array = []
        for _array in _12lead_re:
            full_array += array_by_name[_array].tolist()
        full_array = np.array(full_array)
        return full_array, array_by_name

    def __len__(self):
        return len(self.file_name)
