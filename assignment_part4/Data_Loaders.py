import torch
import torch.utils.data.dataset as dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import pickle
from sklearn.utils import resample

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


class Nav_Dataset(dataset.Dataset):
    def __init__(self, is_train = False):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')

        # Added Functions - Kiran
        self.remove_incorrect_collisions()
        self.prune_dataset(ratio=0.35)

# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data)  # fits and transforms
        temp = pd.DataFrame(self.normalized_data)
        temp.to_csv('normalized_data.csv', header=False, index=False)
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb"))  # save to normalize at inference

        # Concatenate input and labels for the dataset
        # self.dataset = np.concatenate((self.input_data, self.labels.reshape(-1, 1)), axis=1)

    def __len__(self):
# STUDENTS: __len__() returns the length of the dataset
        return len(self.normalized_data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.
        # print(self.normalized_data.shape)

        if idx < len(self):
            input_data = torch.tensor(self.normalized_data[idx, :6], dtype=torch.float32)
            label = torch.tensor(self.normalized_data[idx, 6], dtype=torch.float32)
            return {'input': input_data, 'label': label}
        else:
            raise IndexError("Index out of bounds")

    def remove_incorrect_collisions(self):
        print("Total:", len(self.data))
        duplicates, replications = [], []
        for i in range(len(self.data)):
            if self.data[i, -1] == 1:
                self.data[i - 1, -1] = 1
                duplicates.append(i)
                replications.append(self.data[i - 1])
        self.data = np.array([self.data[i] for i in range(len(self.data)) if i not in duplicates])

    def prune_dataset(self, ratio=0.25):
        self.data = self.data[~(self.data > 150).any(axis=1)]
        non_collisions, collisions = self.data[self.data[:, -1] == 0], self.data[self.data[:, -1] == 1]
        non_collision_count, collision_count = len(non_collisions), len(collisions)
        print("Non-collision Count:", non_collision_count)
        print("Collision Count:", collision_count)

        new_total = collision_count / ratio
        non_collisions_target = int((1 - ratio) * new_total)
        non_collisions_resampled = resample(non_collisions, replace=False, n_samples=non_collisions_target, random_state=123)
        self.data = np.vstack((collisions, non_collisions_resampled))
        np.random.shuffle(self.data)
        print("Final Non-collision Count:", len(self.data[self.data[:, -1] == 0]))
        print("Final Collision Count:", len(self.data[self.data[:, -1] == 1]))


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary
        self.train_dataset, self.test_dataset = train_test_split(self.nav_dataset, test_size = 0.2, random_state=42)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)


def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']


if __name__ == '__main__':
    main()
