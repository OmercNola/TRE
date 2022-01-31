from __future__ import absolute_import, division, print_function
import os
from pathlib import Path, PureWindowsPath, PurePosixPath
from torch.utils.data import Dataset
from datasets_and_loaders.data_preprocess import *
class TRE_train_dataset(Dataset):

    def __init__(self, ):
        super().__init__()

        "=============================================================="
        """TimeBank"""
        TimeBank_folder = Path('./data/TBAQ-cleaned/TimeBank/')
        TimeBank_labeled_data = Path('./data/timebank.txt')
        TimeBank_data_with_markers = final_data_process_for_markers(
            TimeBank_folder, TimeBank_labeled_data
        )
        "=============================================================="
        """Aquaint"""
        Aq_folder = Path('./data/TBAQ-cleaned/AQUAINT/')
        Aq_labeled_data = Path('./data/aquaint.txt')
        Aq_data_with_markers = final_data_process_for_markers(
            Aq_folder, Aq_labeled_data
        )
        "=============================================================="
        """Aquaint and Timebank with markers (train data)"""
        self.TRE_training_data_with_markers = \
            Aq_data_with_markers + TimeBank_data_with_markers
        "=============================================================="

    def __len__(self):
        return len(self.TRE_training_data_with_markers)

    def __getitem__(self, idx):
        res = self.TRE_training_data_with_markers[idx]
        return res
class TRE_val_dataset(Dataset):

    def __init__(self, ):
        super().__init__()

        "=============================================================="
        """TCR (val data)"""
        TCR_folder = Path('./data/TBAQ-cleaned/TemporalPart/')
        self.TRE_validation_data_with_markers = process_TCR_data(TCR_folder)
        "=============================================================="

    def __len__(self):
        return len(self.TRE_validation_data_with_markers)

    def __getitem__(self, idx):
        res = self.TRE_validation_data_with_markers[idx]
        return res
class TRE_test_dataset(Dataset):

    def __init__(self, ):
        super().__init__()

        "=============================================================="
        """Platinum (test data)"""
        Platinum_folder = Path('./data/TBAQ-cleaned/platinum/')
        Platinum_labeled_data = Path('./data/platinum.txt')
        self.TRE_test_data_with_markers = final_data_process_for_markers(
            Platinum_folder, Platinum_labeled_data
        )
        "=============================================================="

    def __len__(self):
        return len(self.TRE_test_data_with_markers)

    def __getitem__(self, idx):
        res = self.TRE_test_data_with_markers[idx]
        return res