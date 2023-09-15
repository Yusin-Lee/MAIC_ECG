import torch
import numpy as np
import pandas as pd


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, df):
        self.file_name = df["FILENAME"].values
        self.gender = df["GENDER"].values
        self.age = df["AGE"].values
        self.file_path = file_path

    def __getitem__(self, index):
        file_name = self.file_name[index]
        full_array = np.load(self.file_path + file_name + ".npy")
        array_by_name = self.array_to_dict_by_name(full_array)
        label = self.age[index]
        return full_array, array_by_name, label

    def __len__(self):
        return len(self.file_name)

    def array_to_dict_by_name(_array):
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
        _dict = dict()
        for name, p_array in zip(_12lead, np.split(_array, 12)):
            _dict[name] = p_array
        return _dict
