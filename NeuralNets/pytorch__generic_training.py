from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch import optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from pytorch__my_vgg import MyVgg  # Trying out my Vgg implementation


def check_harware():
    """
    Empty cache and output the available hardware.

    """
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using {} device: {}".format(device, torch.cuda.get_device_name()))


def load_data(batch_size=32):
    """
    Change this part. By default it just downloads cifar10 and sets it up with DataLoader.

    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_dl = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(trainset, batch_size=batch_size, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    classes_dict = {i: [temp for temp in classes][i] for i, x in enumerate(classes)}

    return (train_dl, val_dl), classes_dict


def train(model, train_dl, val_dl, criterion, optimizer, scheduler, num_epochs=10):
    """
    Given the model, train_DataLoader, val_DataLoader etc start the training loop.
    By default num_epochs is equal to 10.

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if device == "cuda":
        print(f'\nStarting training on {torch.cuda.get_device_name()} [{device}]...\n')
    else:
        print(f'\nStarting training on {device}...\n')

    for epoch in tqdm(range(num_epochs)):

        # Training Part
        running_loss = 0.
        running_corrects = 0
        items_processed = 0
        print(f'Epoch {epoch + 1}/{num_epochs}\n----------')
        for batch, (inputs, labels) in enumerate(train_dl):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)

            # with torch.set_grad_enabled(True):
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            items_processed += labels.shape[0]
            running_loss += loss * labels.shape[0]
            running_corrects += torch.sum(preds == labels)

        epoch_train_loss = running_loss / items_processed
        epoch_train_corrects = running_corrects / items_processed

        print(f'train loss: {epoch_train_loss:.3f} | train acc: {epoch_train_corrects * 100:.2f}%')

        # Validation Part
        running_loss = 0.
        running_corrects = 0
        items_processed = 0
        for batch, (inputs, labels) in enumerate(val_dl):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, dim=1)

                items_processed += labels.shape[0]
                running_loss += loss * labels.shape[0]
                running_corrects += torch.sum(preds == labels)

        epoch_val_loss = running_loss / items_processed
        epoch_val_corrects = running_corrects / items_processed
        print(f'valid loss: {epoch_val_loss:.3f} | valid acc: {epoch_val_corrects * 100:.2f}%')

        scheduler.step()
        print(f'Current lr: {optimizer.state_dict()["param_groups"][0]["lr"]}\n')

    return model, epoch + 1, optimizer.state_dict()


def save_model(file_name, model, optimizer, epoch):
    PATH = 'models/' + file_name

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                }, PATH)


if __name__ == '__main__':
    model = MyVgg(num_classes=10, custom_vgg=[3, 6, 'P'])
    check_harware()
    (train_dl, val_dl), _ = load_data(batch_size=32)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=4, gamma=1)
    model, optimizer_State_dict, epoch = train(model, train_dl, val_dl, criterion, optimizer, scheduler, num_epochs=10)

    save_model('test_model_1.pth', model, optimizer, epoch)
