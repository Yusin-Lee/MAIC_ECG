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
        label = self.age[index]
        return full_array, array_by_name, label

    def __len__(self):
        return len(self.file_name)


def np_merge(df):
    file_name = df["FILENAME"].values
    gender = df["GENDER"].values
    age = df["AGE"].values
    file_path = file_path
