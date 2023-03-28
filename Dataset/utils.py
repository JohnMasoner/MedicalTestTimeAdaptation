import os
import numpy as np

import SimpleITK as sitk


def split_filename(filename: str):
    if filename.endswith(".nii.gz"):
        str_list = filename.split(".nii.gz")
    elif filename.endswith(".mha"):
        str_list = filename.split(".mha")
    elif filename.endswith(".mhd"):
        str_list = filename.split(".mhd")
    elif filename.endswith(".npz"):
        str_list = filename.split(".npz")
    elif filename.endswith(".npy"):
        str_list = filename.split(".npy")
    else:
        str_list = [file_name, ""]
    return str_list


def load_file(filename: str):
    suffix = data_main_modal_path.split(".")[-1]
    if suffix == ".gz" or suffix == ".nii":
        data = sitk.GetArrayFromImage(sitk.ReadImage(filename))
    elif suffix == ".npy" or suffix == ".npz":
        data = np.load(filename)
    else:
        data = np.load(filename + ".npy")

    return data
