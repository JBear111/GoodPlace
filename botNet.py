import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import matplotlib.image as img
import matplotlib.pyplot as plt

PATHSEASON = ""
PATH1 = ""
PATH2 = ""
PATH3 = ""
PATH4 = ""

#model that recognizes episodes of season 1 and 4 because there are 13 episodes in those seasons
class Net1(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(12800, 1024)
        self.fc2 = nn.Linear(1024, 13)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

#model that recognizes episodes of seasons 2 and 3, because there were 12 episodes in those seasons
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(12800, 1024)
        self.fc2 = nn.Linear(1024, 12)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

#model that recognizes what season is the episode from
class NetSezone(nn.Module):
    def __init__(self):
        super(NetSezone, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(12800, 1024)
        self.fc2 = nn.Linear(1024, 4)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

#transfrom pictures, resizes them to 96x96 and normalizes them
train_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize((96,96)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4883, 0.4551, 0.4170), (0.2208, 0.2161, 0.2163))])


#function that recognizes episode, argument is the path of the picture that you want to recognize for
def goodPlace(path):
    #first load model that recognizes what season the episode is from
    device = torch.device('cuda:0')
    model = NetSezone().to(device)
    model.load_state_dict(torch.load(PATHSEASON))
    
    #reads image, transforms it
    picture = img.imread(path)
    picture = train_transform(slika)
    #plt.imshow(transforms.ToPILImage()(slika), interpolation="bicubic")
    #plt.show()
    picture.unsqueeze_(0)

    model.eval()
    picture = picture.to(device)
    outputs = model(picture)
    #rez is the season (integer 1-4)
    rez = int(torch.argmax(outputs.data) + 1)
    if rez == 1 or rez == 4:
        model = Net1().to(device)
        if rez == 1:
            model.load_state_dict(torch.load(PATH1))
        else:
            model.load_state_dict(torch.load(PATH4))
    else:
        model = Net2().to(device)
        if rez == 2:
            model.load_state_dict(torch.load(PATH2))
        else:
            model.load_state_dict(torch.load(PATH3))
    picture = picture.to(device)
    outputs = model(picture)
    ep = int(torch.argmax(outputs.data) + 1)
    #returns string in format S{}-ep{}.jpg, so you can rename the pictures
    return "S{}-ep{}.jpg".format(rez,ep)

if __name__ == "__main__":
    path = ""
    goodPlace(path)
