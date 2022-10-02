import torch
import torch.nn as nn



class BasicConvolutionNeuralNetwork(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.opt.channels.insert(0,1) # insert number of channel of MNIST is one into the first channels list
        self.layers = nn.ModuleList()
        for i in range(len(self.opt.channels) - 1):
            self.layers.append(nn.Conv2d(self.opt.channels[i], self.opt.channels[i+1], (self.opt.kernel_size, self.opt.kernel_size),padding=0, stride=1))
            self.layers.append(nn.MaxPool2d(self.opt.maxpool_kernel_size))

        self.layers.append(nn.Flatten())
#        print(self.opt.dense_size, self.get_size_after_flatten())
        self.pred_layers = nn.Sequential(
            nn.Linear(int(self.get_size_after_flatten()), self.opt.dense_size),
            nn.ReLU(),
            nn.Dropout(self.opt.dropout),
            nn.Linear(self.opt.dense_size, self.opt.num_classes),
            nn.Softmax(),
        )
    def get_size_after_flatten(self):
        sample_image = torch.rand((self.opt.batch_size,1,28,28))
        out = sample_image
        for layer in self.layers:
            out = layer(out)
#        print(out.reshape(self.opt.batch_size, -1).shape)
        return out.reshape(self.opt.batch_size, -1).shape[-1]
            
    def forward(self,x):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.pred_layers(out)
        return out
