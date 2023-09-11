def main():
    import os
    import time
    import copy
    from glob import glob
    import cv2
    import shutil

    import numpy as np
    import torch
    from torch import device
    import torchvision
    import torchvision.transforms as transforms
    import torchvision.models as models
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    import matplotlib.pyplot as plt

    data_path = "../data/cat_and_dog/train"

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, num_workers=8, shuffle=True)

    print("CUDA is available : ", torch.cuda.is_available())
    print("GPU count : ", torch.cuda.device_count())
    for gpu in range(torch.cuda.device_count()):
        print("GPU name : ", torch.cuda.get_device_name(gpu))
    print("Length of train_dataset : ", len(train_dataset))

    samples, labels = next(iter(train_loader))

    classes = {0: "cat", 1: "dog"}
    fig = plt.figure(figsize=(16, 24))

    for i in range(24):
        a = fig.add_subplot(4, 6, i + 1)
        a.set_title(classes[labels[i].item()])
        a.axis("off")
        a.imshow(np.transpose(samples[i], (1, 2, 0)))
    plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)

    resnet = models.resnet18(pretrained=True)

    def set_parameter_requires_grad(_model, feature_extract=True):
        if feature_extract:
            for _param in _model.parameters():
                _param.requires_grad = False

    set_parameter_requires_grad(resnet)

    resnet.fc = nn.Linear(512, 2)

    for name, param in resnet.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(512, 2)
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.fc.parameters())
    cost = nn.CrossEntropyLoss()

    print(model)

    def train_model(mod, dataloaders, crit, opti, _device, num_epochs=13):
        since = time.time()
        acc_history = []
        loss_history = []
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print("-" * 10)

            running_loss = 0.0
            running_corrects = 0

            for inputs_, labels_ in dataloaders:
                inputs_, labels_ = inputs_.to(_device), labels_.to(_device)

                mod.to(_device)
                opti.zero_grad()
                outputs = mod(inputs_)
                loss = crit(outputs, labels_)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                opti.step()

                running_loss += loss.item() * inputs_.size(0)
                running_corrects += torch.sum(preds == labels_.data)

            epoch_loss = running_loss / len(dataloaders.dataset)
            epoch_acc = running_corrects.double() / len(dataloaders.dataset)

            print("Loss : {:.4f} Acc : {:.4f}".format(epoch_loss, epoch_acc))

            if epoch_acc > best_acc:
                best_acc = epoch_acc

            acc_history.append(epoch_acc.item())
            loss_history.append(epoch_loss)

            if not os.path.isdir("../data/model"):
                os.mkdir("../data/model")

            torch.save(mod.state_dict(), os.path.join("../data/model", "{0:0=2d}.pth".format(epoch)))
            print()

        time_elapsed = time.time() - since
        print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
        print("Best Acc : {:.4f}".format(best_acc))
        return acc_history, loss_history

    params_to_update = []
    for name, param in resnet.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t", name)

    device = device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    train_acc_hist, train_loss_hist = train_model(resnet, train_loader, criterion, optimizer, device)

    test_path = "../data/cat_and_dog/test"

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)

    test_loader = DataLoader(dataset=test_dataset, batch_size=32, num_workers=1, shuffle=True)

    print("Length of test_dataset : ", len(test_dataset))

    def eval_model(model,dataloaders, device):
        since = time.time()
        acc_history = []
        best_acc = 0.0

        saved_models = glob("../data/model/*.pth")
        saved_models.sort()
        print("Saved models : ", saved_models)

        for model_path in saved_models:
            print("Load model : ", model_path)

            model.load_state_dict(torch.load(model_path))
            model.eval()
            model.to(device)
            running_corrects = 0

            for inputs, labels in dataloaders:
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)

                _, preds = torch.max(outputs, 1)
                preds[preds >= 0.5] = 1
                preds[preds < 0.5] = 0
                running_corrects += preds.eq(labels.data.view_as(preds)).sum()

            epoch_acc = running_corrects.double() / len(dataloaders.dataset)
            print("Acc : {:.4f}".format(epoch_acc))

            if epoch_acc > best_acc:
                best_acc = epoch_acc

            acc_history.append(epoch_acc.item())
            print()

        time_elapsed = time.time() - since
        print("Eval complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
        print("Best Acc : {:.4f}".format(best_acc))

        return acc_history

    val_acc_hist = eval_model(resnet, test_loader, device)

    plt.plot(train_acc_hist, label="Train Acc")
    plt.plot(val_acc_hist, label="Test Acc")
    plt.show()

    plt.plot(train_loss_hist, label="Train Loss")
    plt.show()


if __name__ == "__main__":
    main()
