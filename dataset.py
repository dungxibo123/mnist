

import torch
import torchvision
import random
from sklearn.model_selection import train_test_split
import copy
import numpy as np

def get_dataset(opt):
    if opt.data == "MNIST":
        dataset = torchvision.datasets.MNIST(root=opt.data_path, download=True,transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
    elif opt.data == "FMNIST":
        dataset = torchvision.datasets.FashionMNIST(root=opt.data_path, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

    if opt.unbalanced:
        num_classes = 10
        classe_labels = range(num_classes)
        sample_probs = torch.rand(num_classes)
       
        idx_to_del = [i for i, label in enumerate(dataset.train_labels) 
                      if random.random() > sample_probs[label]]
        imbalanced_train_dataset = copy.deepcopy(dataset)
        
        imbalanced_train_dataset.train_labels = np.delete(dataset.train_labels, idx_to_del, axis=0)
        imbalanced_train_dataset.train_data = np.delete(dataset.train_data, idx_to_del, axis=0) 
    
        return imbalanced_train_dataset
    return dataset

def get_train_val_loader(opt):
    dataset = get_dataset(opt)
    train_set, val_set = train_test_split(dataset, test_size=opt.test_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=opt.batch_size, num_workers=opt.workers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set , shuffle=True, batch_size=opt.val_batch_size, num_workers=opt.workers, drop_last=True)

    return train_loader, val_loader



if __name__ == "__main__":
    #opt = {"data": "MNIST", "data_path": "./data", "test_size": 0.3, "batch_size": 32, "val_batch_size": 32, "workers": 2}
    #get_train_val_loader(opt)
    pass
