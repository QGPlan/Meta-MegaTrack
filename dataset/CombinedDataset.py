import torch

from HMDHandRectsDataset import HMDHandRectsDataset
from EpicKitchensDataset import EpicKitchensDataset
from DarknetDataset import DarknetDataset


import nonechucks


class RepeatDataset(torch.utils.data.Dataset):
    def __init__(self, wrap_dataset, num_times):
        self.wrap_dataset = wrap_dataset

        self.num_times = num_times
        self.actual_size = len(self.wrap_dataset)

    def __len__(self):
        return self.actual_size*self.num_times

    def __getitem__(self, idx):
        return self.wrap_dataset[idx % self.actual_size]


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self):
        amts = []
        datasets = []

        def b(ds, am):
            amts.append(am)
            datasets.append(ds)

        hmdhandrect_datasets = []

        hmdhandrect_datasets.append(HMDHandRectsDataset(
            "datasets/HMDHandRects/sequences/train_subject00_sequence00"))

        hmdhandrect_datasets.append(HMDHandRectsDataset(
            "datasets/HMDHandRects/sequences/train_subject00_sequence01"))

        hmdhandrect_datasets.append(HMDHandRectsDataset(
            "datasets/HMDHandRects/sequences/train_subject00_sequence02"))

        hmdhandrect_datasets.append(HMDHandRectsDataset(
            "datasets/HMDHandRects/sequences/train_subject00_sequence03"))

        hmdhandrect_datasets.append(HMDHandRectsDataset(
            "datasets/HMDHandRects/sequences/train_subject01_sequence00"))

        hmdhandrect_datasets.append(HMDHandRectsDataset(
            "datasets/HMDHandRects/sequences/train_subject01_sequence01"))

        hmdhandrect_datasets.append(HMDHandRectsDataset(
            "datasets/HMDHandRects/sequences/train_subject02_sequence00"))

        hmdhandrect_datasets.append(HMDHandRectsDataset(
            "datasets/HMDHandRects/sequences/train_subject02_sequence01"))

        b(torch.utils.data.ConcatDataset(hmdhandrect_datasets), 2)

        b(DarknetDataset(
            "/excluded_epics_from_sync/initial_frontend_T32624/hand-convert/ego-hands/"), .5)

        b(EpicKitchensDataset(), 5)

        repeat_datasets = []

        biggest_dataset_len = 0
        biggest_dataset_associated_amt = 0
        amt_sum = 0
        for amt, ds in zip(amts, datasets):
            ds_size = len(ds)
            print("sizer", ds_size)
            if ds_size > biggest_dataset_len:
                biggest_dataset_len = ds_size
                biggest_dataset_associated_amt = amt
            amt_sum += amt

        for amt, ds in zip(amts, datasets):
            num_times = int((amt/biggest_dataset_associated_amt)
                            * (biggest_dataset_len/len(ds)))
            # ds.num_times_to_repeat = num_times
            repeat_datasets.append(RepeatDataset(ds, num_times))
            # try:
            #   ds.dataset.num_times_to_repeat = num_times
            # except:
            #   print("no")
            print("num", num_times)

        self.ds = torch.utils.data.ConcatDataset(repeat_datasets)
        # raise

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


if __name__ == '__main__':

    a = CombinedDataset()
    raise
