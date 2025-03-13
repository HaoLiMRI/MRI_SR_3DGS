"""The code to organize the MRI SR target domain dataset, which contains only the measured MRI LR data."""

# import gzip
# import os
# import pickle
# import urllib

import numpy as np
import torch as tc
# import torch.utils.data as data
# from torchvision import datasets, transforms
import h5py


def EVAL_LOADER(root):
    """
    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

#    def __init__(self, root):
    """Init USPS dataset."""
        # init params
#    self.root = os.path.expanduser(root)
#    self.forward()
#    def forward(self):
    print(root)
    print('One more low resolution image set exist')
    file_data = h5py.File(root, 'r')
    data_low_resolution = file_data['HRGT'][:] #----- numpy array
    print(np.shape(data_low_resolution))
    print(data_low_resolution.dtype)
    torch_data_low_resolution = tc.from_numpy(data_low_resolution) #----- torch type data could be read by tc.utils.data.TensorDataset
    "Note the original .mat file has 2D image matrix by number_of_data_samples, which is H x W x N. After reading into h5py, the dimension changes as N x W x H. However in 2D MRI SR, so we have to permute axis to form the data on N x H x W"
    torch_data_low_resolution = torch_data_low_resolution.permute(0, 1, 3, 2)
    print(np.shape(torch_data_low_resolution))
    torch_data_low_resolution_sequence = torch_data_low_resolution
    print(np.shape(torch_data_low_resolution_sequence))
    torch_data_low_resolution_sequence = torch_data_low_resolution_sequence.float()
    print(np.shape(torch_data_low_resolution_sequence))
    return tc.utils.data.TensorDataset(torch_data_low_resolution_sequence)


def get_dasr_hr_loader(root, batch_size, shuffle=False):
    """Get Target dataset loader."""
    # dataset and data loader
    evaluation_dataset = EVAL_LOADER(root)

    evaluation_loader = tc.utils.data.DataLoader(
        dataset=evaluation_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=shuffle)

        
    return evaluation_loader
