from __future__ import absolute_import, division, print_function
import os
from data.data_preprocess import *
from torch.utils.data import Dataset
import random
from pathlib import Path, PureWindowsPath, PurePosixPath
class TRE_train_dataset(Dataset):

    def __init__(self, args):
        super().__init__()
        random.seed(args.seed)
        "=============================================================="
        """TimeBank"""
        TimeBank_folder = Path('./raw_data/TBAQ-cleaned/TimeBank/')
        TimeBank_labeled_data = Path('./raw_data/timebank.txt')
        TimeBank_data_with_markers = final_data_process_for_markers(
            args, TimeBank_folder, TimeBank_labeled_data
        )
        "=============================================================="
        """Aquaint"""
        Aq_folder = Path('./raw_data/TBAQ-cleaned/AQUAINT/')
        Aq_labeled_data = Path('./raw_data/aquaint.txt')
        Aq_data_with_markers = final_data_process_for_markers(
            args, Aq_folder, Aq_labeled_data
        )
        "=============================================================="
        """Aquaint and Timebank with markers (train raw_data)"""
        self.TRE_training_data_with_markers = \
            Aq_data_with_markers + TimeBank_data_with_markers

        # take sample without replacment:
        assert (0 <= args.part_of_train_data <= 12736)
        if args.part_of_train_data == 0:
            self.TRE_training_data_with_markers = []
        else:
            self.TRE_training_data_with_markers = \
                random.sample(self.TRE_training_data_with_markers, args.part_of_train_data)
            if args.rank == 0:
                print(self.TRE_training_data_with_markers[0])
        "=============================================================="

    def __len__(self):
        return len(self.TRE_training_data_with_markers)

    def __getitem__(self, idx):
        res = self.TRE_training_data_with_markers[idx]
        return res
class TRE_val_dataset(Dataset):

    def __init__(self, args):
        super().__init__()

        "=============================================================="
        """TCR (val raw_data)"""
        TCR_folder = Path('./raw_data/TBAQ-cleaned/TemporalPart/')
        self.TRE_validation_data_with_markers = process_TCR_data(args, TCR_folder)
        "=============================================================="

    def __len__(self):
        return len(self.TRE_validation_data_with_markers)

    def __getitem__(self, idx):
        res = self.TRE_validation_data_with_markers[idx]
        return res
class TRE_test_dataset(Dataset):

    def __init__(self, args):
        super().__init__()

        "=============================================================="
        """Platinum (test raw_data)"""
        Platinum_folder = Path('./raw_data/TBAQ-cleaned/platinum/')
        Platinum_labeled_data = Path('./raw_data/platinum.txt')
        self.TRE_test_data_with_markers = final_data_process_for_markers(
            args, Platinum_folder, Platinum_labeled_data
        )
        "=============================================================="

    def __len__(self):
        return len(self.TRE_test_data_with_markers)

    def __getitem__(self, idx):
        res = self.TRE_test_data_with_markers[idx]
        return res