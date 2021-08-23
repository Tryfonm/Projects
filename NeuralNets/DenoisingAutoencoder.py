# %%

from tqdm import trange

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# ==================== Load Mnist ==================== #
batch_size = 64

train_dataset = datasets.MNIST(root='./', download=True, train=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./', download=True, train=False, transform=transforms.ToTensor())

x_train = train_dataset.data.float()
y_train = train_dataset.targets
y_train_ohe = torch.nn.functional.one_hot(y_train)

train_dl = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)
test_dl = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)


model = DenoisingAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()
max_epochs = 5
noise_level = 0.75
# ==================== Train ==================== #

training_losses = []
for i in (t := trange(max_epochs, ncols= 100)):
    for images, labels in train_dl:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Add noise to the images by blacking out pixels
        noisy_images = nn.Dropout(p=noise_level)(images)
        reconstructions = model(noisy_images)

        model.zero_grad()
        loss = criterion(reconstructions, images)
        training_losses.append(loss.item())
        loss.backward()

        optimizer.step()

        t.set_description(f'Training loss: {loss.item():.4f}')

plt.plot(training_losses, '.-'); plt.show()

# ==================== Check results ==================== #

model = model.to('cpu')

batch_index = 0
number_of_images = 10
images = next(iter(train_dl))[batch_index][0:number_of_images]

# Original image
fig, axis = plt.subplots(3, number_of_images, figsize=(13, 4))
for index, image in enumerate(images):
    axis[0, index].imshow(image.permute(1, 2, 0))
    axis[0, index].set_axis_off()
    axis[0, index].set_aspect('equal', 'box')
plt.subplots_adjust(wspace=0, hspace=0)

# Reconstruction
with torch.no_grad():
    images = next(iter(train_dl))[0][0:number_of_images]
    noisy_images = nn.Dropout(p=0.50)(images.reshape(-1, 28*28))
    for index, img in enumerate(noisy_images):
        reconstructed_image = model(img)
        axis[1, index].imshow(noisy_images[index].reshape(28,28,1))
        axis[2, index].imshow(reconstructed_image.reshape(28, 28, 1))
        axis[1, index].set_axis_off()
        axis[2, index].set_axis_off()
        axis[1, index].set_aspect('equal', 'box')
        axis[2, index].set_aspect('equal', 'box')
plt.show()