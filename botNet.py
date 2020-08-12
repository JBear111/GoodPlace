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

class DCDataset(Dataset):
    def __init__(self, data, path, transform=None):
        super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, label = self.data[index]
        img_path = os.path.join(self.path, img_name)
        image = img.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(12800, 1024)
        self.fc2 = nn.Linear(1024, 13)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        #self.conv2_drop = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(12800, 1024)
        self.fc2 = nn.Linear(1024, 12)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class NetSezone(nn.Module):
    def __init__(self):
        super(NetSezone, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        #self.conv2_drop = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(12800, 1024)
        self.fc2 = nn.Linear(1024, 4)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

train_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize((96,96)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4883, 0.4551, 0.4170), (0.2208, 0.2161, 0.2163))])


def goodPlace(path):
    device = torch.device('cuda:0')
    model = NetSezone().to(device)
    model.load_state_dict(torch.load("sezone.pth"))

    slika = img.imread(path)
    slika = train_transform(slika)
    plt.imshow(transforms.ToPILImage()(slika), interpolation="bicubic")
    plt.show()
    slika.unsqueeze_(0)

    model.eval()
    slika = slika.to(device)
    outputs = model(slika)
    rez = int(torch.argmax(outputs.data) + 1)
    if rez == 1 or rez == 4:
        model = Net1().to(device)
        if rez == 1:
            model.load_state_dict(torch.load("season1.pth"))
        else:
            model.load_state_dict(torch.load("season4.pth"))
    else:
        model = Net2().to(device)
        if rez == 2:
            model.load_state_dict(torch.load("season2.pth"))
        else:
            model.load_state_dict(torch.load("season3.pth"))
    slika = slika.to(device)
    outputs = model(slika)
    ep = int(torch.argmax(outputs.data) + 1)
    return "S{}-ep{}.jpg".format(rez,ep)

if __name__ == "__main__":
    path = "Reddit/pnok4fof8kf51.jpg"
    goodPlace(path)