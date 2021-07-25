import os
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
- An optimizer simply changes the model parameters to reduce the loss
- So they are problem agnostic, any optimizer will work. Some just perform better than others depending
  on how difficult the optimization is.
"""
from torch.optim import Adam  # only optimizer I could find for binary classification
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 512
LEARNING_RATE = 0.001
EPOCHS = 2

"""
- Lets go through this in our next meeting. It makes a lot more sense than what it looks, see some comments inline below
- For pytorch datasets you need to write a __getitem__ method that accepts an index to tell the pytorch dataloaded how
  to index into your data. Similarly you need to write a __len__ method so that the dataloader knows how big the dataset
  is and when to stop. They are required methods that get called internally by dataloader, which is very powerful
  because now we can write arbitrary dataloaders using plain python
"""


class ImageDataset(Dataset):
    def __init__(self, dirname):
        super(Dataset).__init__()
        self.dirname = dirname  # Accept a directory to read images from
        self.images = os.listdir(self.dirname)  # Get list of all files in above directory

    def __getitem__(self, index):
        image_name = self.images[index]  # Select image from image list in index
        image = read_image(os.path.join(self.dirname, image_name))  # Read image from disk into matrix
        if image_name.split("_")[0] == "photo":  # If name of file contains "photo" we make label zero
            label = torch.tensor([0])
        else:  # All other filenames start with "monet" so make label 1
            label = torch.tensor([1])
        return image, label

    def __len__(self):
        return len(self.images)


train_dataset = ImageDataset("./data/train")
validation_dataset = ImageDataset("./data/validate")
test_dataset = ImageDataset("./data/test")

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        """
        - This is correct, we know it will perform poorly
        - The last layer in the network should have a single node for binary classification
        """
        self.layer_1 = nn.Linear(in_features=256 * 256 * 3, out_features=1)  # ???

    def forward(self, x):
        x = self.layer_1(x)
        """
        - Because we only have one hidden layer in the network, we do not need the relu as the sigmoid will be the
          activation function.
        - This will change when we add more layers.
        """
        # x = F.relu(x)
        """
        - softmax is for multiclass classification
        - for binary classification last layer actication should be a sigmoid
        """
        # output = F.log_softmax(x, dim=1)  # ??? dim????
        output = F.sigmoid(x)
        return output


def train(model, optimizer):
    for epoch in range(EPOCHS):
        for batch_number, (image, label) in enumerate(train_dataloader):
            image = image.view(-1, 256 * 256 * 3).to(torch.float32)
            """
            - Because image is cast to float32, need to cast label to float32 as well for the loss calculation
            """
            label = label.to(torch.float32)
            prediction = model(image)
            """
            - In below loss there was a syntax error (missing bracket)
            """
            loss = F.binary_cross_entropy(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step

            if batch_number % 2 == 0:
                print(f"Epoch: {epoch} \t |Batch: {batch_number} \t | Loss: {loss}")


if __name__ == "__main__":
    model = BinaryClassifier()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, optimizer)
    # for batch_number, (image, label) in enumerate(train_dataloader):
    #     print(image.size()) # view???? -> turns tensor into different shape
    #     break
