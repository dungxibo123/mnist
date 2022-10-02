

import torch
import torchvision
from sklearn.model_selection import train_test_split

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
