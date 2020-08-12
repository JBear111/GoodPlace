import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.image as img


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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(12800, 1024)
        self.fc2 = nn.Linear(1024, 13)

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


categories = []
filename = os.listdir(PATH)
for file in filename:
    category = file.split("-")[1]
    categories.append(int(category[2:]) - 1)

df = pd.DataFrame({
    'filename': filename,
    'category': categories
})

train_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize(96),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4883, 0.4551, 0.4170), (0.2208, 0.2161, 0.2163))
                                      ])

train_data = DCDataset(df, PATH, train_transform)

epochs = 70
classes = 2
batch = 35
learning_rate = 0.001


train_loader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True, num_workers=0)

device = torch.device('cuda')
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate, momentum=0.9)

train_losses = []
valid_losses = []

for epoch in range(1, epochs + 1):
    # keep-track-of-training-and-validation-loss
    train_loss = 0.0
    valid_loss = 0.0
    correct_train = 0
    correct_valid = 0

    # training-the-model
    model.train()
    for data, target in train_loader:


        # move-tensors-to-GPU
        data = data.to(device)
        target = target.to(device)

        # clear-the-gradients-of-all-optimized-variables
        optimizer.zero_grad()
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        correct_train += (predicted == target).sum().item()
        # calculate-the-batch-loss
        loss = criterion(output, target)
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        optimizer.step()
        # update-training-loss
        train_loss += loss.item() * data.size(0)

    # calculate-average-losses
    train_loss = train_loss / len(train_loader.sampler)
    correct_train = correct_train / len(train_loader.sampler)
    train_losses.append(train_loss)
    correct_valid = 0
    valid_loss = 0

    # print-training/validation-statistics
    print(
        'Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tCorrect training: {} \tCorrect valid: {}'.format(
            epoch, train_loss, valid_loss, correct_train, correct_valid))


PATH = ""
torch.save(model.state_dict(), PATH)
