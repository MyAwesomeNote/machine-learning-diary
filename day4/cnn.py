import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.nn import functional as nn_func
from torch.utils.data import DataLoader

print("CUDA is", "Enabled" if torch.cuda.is_available() else "Not enabled")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = torchvision.datasets.FashionMNIST("../data/MNIST", download=True,
                                                  transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.FashionMNIST("../data/MNIST", download=True, train=False,
                                                 transform=torchvision.transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

labels_map = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
              5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}

fig = plt.figure(figsize=(8, 8))
columns = 4
rows = 5
for i in range(1, columns * rows + 1):
    img_xy = np.random.randint(len(train_dataset))
    img = train_dataset[img_xy][0][0, :, :]
    fig.add_subplot(rows, columns, i)
    plt.title(labels_map[train_dataset[img_xy][1]])
    plt.axis("off")
    plt.imshow(img, cmap="gray")

plt.show()


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(int(32 * (48 + 1) / 2), 600)  # 784
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(600, 120)
        self.fc3 = nn.Linear(120, 10)

    def forward(self, input_data):
        out = self.layer1(input_data)
        out = self.layer2(out)
        out = input_data.view(out.size(0), -1)
        out = nn_func.relu(self.fc1(out))
        out = self.drop(out)
        out = nn_func.relu(self.fc2(out))
        out = self.fc3(out)
        return out


learning_rate = 0.001
model = FashionCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(model)

num_epochs = 100
count = 0

loss_list = []
iteration_list = []
accuracy_list = []

predictions_list = []
labels_list = []

for epoch in range(num_epochs):
    for i, (images, train_labels) in enumerate(train_loader):
        images, train_labels = images.to(device), train_labels.to(device)

        train = Variable(images.view(100, 1, 28, 28))
        train_labels = Variable(train_labels)

        outputs = model(train)
        loss = criterion(outputs, train_labels)

        optimizer.zero_grad()  # 기울기 초기화
        loss.backward()  # 역전파
        optimizer.step()  # 가중치 갱신

        count += 1

        if not (count % 50):
            total = 0
            correct = 0

            for image, labels in test_loader:
                image, labels = image.to(device), labels.to(device)
                labels_list.append(labels)

                test = Variable(image.view(100, 1, 28, 28))
                outputs = model(test)

                predictions = torch.max(outputs, 1)[1].to(device)
                correct += (predictions == labels).sum()
                total += len(labels)

            accuracy = correct * 100 / total

            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)

            if not (count % 100):
                msg = "Iteration: {}, Loss: {}, Accuracy: {}%"
                print(('★ ' + msg if count % 500 == 0 else msg).format(count, loss.data, accuracy))
