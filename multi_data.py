import random

import numpy as np
from torch.utils.data import Dataset


class Multi_view_data(Dataset):
    """
    A PyTorch Dataset class for handling multi-view data.
    """

    def __init__(self, view_number, idx, feature_list, labels):
        """
        Initializes the Multi_view_data object.

        Parameters:
        - view_number: the number of different views or modalities in the dataset.
        - idx: indices of the samples to include in the dataset.
        - feature_list: a list containing feature arrays for each view.
        - labels: an array of labels corresponding to the samples.
        """
        super(Multi_view_data, self).__init__()
        self.X = dict()
        for v_num in range(view_number):
            self.X[v_num] = feature_list[v_num][[idx], :].squeeze()
        self.y = labels[idx].astype(np.int64)

    def __getitem__(self, index):
        data = dict()
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
        target = self.y[index]
        return (data, target)

    def __len__(self):
        return len(self.X[0])


class Multi_view_OMix_data(Dataset):
    """
    A PyTorch Dataset for handling multi-view data with the O-Mix data augmentation technique.
    """

    def __init__(self, view_number, idx, feature_list, labels, c=0.5, alpha=1.5):
        """
        Initializes the dataset object.
        :param view_number: Number of views in the dataset.
        :param idx: Indices of samples to include.
        :param feature_list: List of features for each view.
        :param labels: Array of labels corresponding to the features.
        :param c: Parameter to adjust the mixup contribution.
        :param alpha: Alpha parameter for the beta distribution used in mixup.
        """
        super(Multi_view_OMix_data, self).__init__()

        self.X = dict()
        self.c = c
        self.alpha = alpha
        for v_num in range(view_number):
            self.X[v_num] = feature_list[v_num][[idx], :].squeeze()
        self.y = labels[idx].astype(np.int64)

        self.class_indices = {label: np.where(self.y == label)[0] for label in np.unique(self.y)}

    def mixup_data(self, x, current_label):
        mixed_x, mixed_y, lambdas, u = {}, {}, [], []

        for i in range(len(x)):
            # 获取不同类别的索引
            different_class_labels = [lbl for lbl in self.class_indices.keys() if lbl != current_label]
            selected_label = random.choice(different_class_labels)
            index = random.choice(self.class_indices[selected_label])  # 确保不同类别

            rf = np.random.beta(self.alpha, self.alpha)
            u.append(self.c * (1 - abs(rf - 0.5)))
            lambdas.append(rf)

            mixed_x[i] = rf * x[i] + (1 - rf) * self.X[i][index]
            mixed_x[i] = mixed_x[i].astype(np.float32)
            mixed_y[i] = self.y[index]

        return mixed_x, mixed_y, lambdas, u

    def __getitem__(self, index):
        data = dict()
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
        target = self.y[index]
        x_mix, Y_mix, lambdas, u = self.mixup_data(data, target)

        return (data, target, x_mix, Y_mix, lambdas, u)

    def __len__(self):
        return len(self.X[0])
#
