from model import BasicConvolutionNeuralNetwork
import utils

import torch
from PIL import Image

import argparse
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
#import transforms

args = argparse.ArgumentParser()
args.add_argument("--model-path", type=str, default="./model/final_model.pt")
args.add_argument("--img", type=str, required=True)

opt = args.parse_args()
state_dict = torch.load(opt.model_path) 
print(state_dict["opt"])

model = BasicConvolutionNeuralNetwork(state_dict["opt"], inferrence=True)
model.load_state_dict(state_dict["model_state_dict"])
img = Image.open(opt.img)
transform=transforms.Compose([
    transforms.Resize((28,28),interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
    (0.1307,),(0.3801,))
])
img = transform(img)
img = img[0,:,:] + img[1,:,:] + img[2,:,:]
img = img / 3
img = img.reshape((1,28,28))
inp_img = img.reshape(1,1,28,28)

out = torch.argmax(model(inp_img)).item()
print(out)
plt.imshow(img.reshape((28,28)), cmap="gray")
plt.xlabel(f"The predicted label is {out}")
plt.show()
