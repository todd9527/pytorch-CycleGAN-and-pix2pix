from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision
import os
import pydicom
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from PIL import Image
import nibabel as nib
import numpy as np


class AmyloidFDGDataset(Dataset):

    def __init__(self, tracer=None, data_dir="/Users/todd/code/research/cafn/data/lowdose-pet-fdg-nifti"):
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5,), (0.5,))
        ])

        # Note: the indices of low_dose_filenames and full_dose_filenames should be paired
        self.low_dose_filenames = []
        self.full_dose_filenames = []
        self.DATA_DIR = data_dir
        self.LOW_DOSE_FILENAME = "803_not_diagnostic_hundredth_dose_recon_mac.nii"
        self.FULL_DOSE_FILENAME = "800_not_diagnostic_full_dose_recon_mac.nii"
        self.SLICES_PER_SCAN = 89
        for scan_dir in os.listdir(self.DATA_DIR):
            self.low_dose_filenames.append(os.path.join(self.DATA_DIR, scan_dir, self.LOW_DOSE_FILENAME))
            self.full_dose_filenames.append(os.path.join(self.DATA_DIR, scan_dir, self.FULL_DOSE_FILENAME))

        # self.LOW_DOSE_FOLDER_NAME = "DRF001" # 1% dose
        # self.FULL_DOSE_FOLDER_NAME = "DRF100"
        # #for patient_dir in os.listdir(self.DATA_DIR):
        # for dicom_filename in os.listdir(self.DATA_DIR + "/" + self.LOW_DOSE_FOLDER_NAME):
        #     self.low_dose_filenames.append(self.DATA_DIR + "/" + self.LOW_DOSE_FOLDER_NAME + "/" + dicom_filename)
        # for dicom_filename in os.listdir(self.DATA_DIR + "/" + self.FULL_DOSE_FOLDER_NAME):
        #     self.full_dose_filenames.append(self.DATA_DIR + "/" + self.FULL_DOSE_FOLDER_NAME + "/" + dicom_filename)

    def __getitem__(self, i):
        scan_index = i // self.SLICES_PER_SCAN
        slice_index = i % self.SLICES_PER_SCAN
        # todo: write code to verify images are paired (looking at dicom metadata)
        # low_dose_img = Image.fromarray(pydicom.read_file(self.low_dose_filenames[i]).pixel_array, mode="L")
        # full_dose_img = Image.fromarray(pydicom.read_file(self.full_dose_filenames[i]).pixel_array, mode="L")
        low_dose_img = nib.load(self.low_dose_filenames[scan_index])._data[:, :, slice_index, np.newaxis]
        full_dose_img = nib.load(self.full_dose_filenames[scan_index])._data[:, :, slice_index, np.newaxis]
        # return self.transforms(low_dose_img), self.transforms(full_dose_img)
        return {'A': low_dose_img, 'B': low_dose_img, 'A_paths': self.low_dose_filenames[scan_index],
                'B_paths': self.full_dose_filenames[scan_index]}

    def __len__(self):
        return len(self.low_dose_filenames) * self.SLICES_PER_SCAN

# dataset = AmyloidFDGDataset()
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# count = 0
# for batch in dataloader:
#       plt.title("LOW DOSE")
#       plt.imshow(batch[0][0][0], cmap="gray")
#       plt.show()
#       plt.title("HIGH DOSE")
#       plt.imshow(batch[0][1][0], cmap="gray")
#       plt.show()
#       count += 1
#       if count == 5:
#           break
