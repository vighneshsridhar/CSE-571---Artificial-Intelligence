import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
from torch.utils.data import DataLoader
import numpy as np
# import pandas as pd
import pickle

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


class Nav_Dataset(dataset.Dataset):
    def __init__(self, is_train):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
        #print("Total:")
        #print(len(self.data))
        #print("Amount of no collisions:")
        #print(len(self.data[self.data[:,-1] == 0]))
        #print("Amount of collisions:")
        #print(len(self.data[self.data[:,-1] == 1]))
# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data

        self.labels = self.data[:, 6]

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        #print(self.normalized_data[0].astype('float32'))
        #print(self.normalized_data[0,:-1].astype('float32'))
        #print(self.normalized_data[0,-1].astype('float32'))
        self.input_data = self.normalized_data[:, 0:5]
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

        if is_train:
            # Split the dataset into training and testing sets
            self.input_data, _, self.labels, _ = train_test_split(self.normalized_data, self.labels, test_size = 0.2, random_state=42)
        else:
            _, self.input_data, _, self.labels = train_test_split(self.normalized_data, self.labels, test_size = 0.2, random_state=42)

        # df = pd.DataFrame(self.input_data)

        #if (is_train):
            #df.to_csv('saved/train.csv')

        #else:
            #df.to_csv('saved/test.csv')

        # Concatenate input and labels for the dataset
        self.dataset = np.concatenate((self.input_data, self.labels.reshape(-1, 1)), axis=1)

    def __len__(self):
# STUDENTS: __len__() returns the length of the dataset
        return len(self.dataset)
        pass

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.
        # print(self.normalized_data.shape)

        data = self.normalized_data
        if idx < len(self):
            input_data = torch.tensor(data[idx, :6], dtype=torch.float32)
            label = torch.tensor(data[idx, 6], dtype=torch.float32)
            return {'input': input_data, 'label': label}
        else:
            raise IndexError("Index out of bounds")


class Data_Loaders():
    def __init__(self, batch_size):
        # self.nav_dataset = Nav_Dataset()
# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary
        self.train_dataset = Nav_Dataset(is_train = True)
        self.test_dataset = Nav_Dataset(is_train = False)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)



        # self.train_loader = [dict(input=row[0:5], label=row[6]) for row in train_df
        # self.test_loader = [dict(input=row[0:5], label=row[6]) for row in test_df[1]]

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
        # print ("train sample input shape = ", sample['input'].shape, "train sample label shape= ", sample['label'].shape)
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']
        # print ("test sample input shape = ", sample['input'].shape, "test sample label shape= ", sample['label'].shape)



if __name__ == '__main__':
    main()
