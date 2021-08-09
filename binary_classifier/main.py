import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize

"""
DC (2021-07-24) - An optimizer simply changes the model parameters to reduce the loss
CW (2021-07-25) - When you say changes the model parameters, do you mean adjusts the dimension
    (i.e. adds columns that converts the labels into higher dimensions like support vector machines)
    or is it just normalizing the numerical data to help it perform faster calculations?
DC (2021-07-26) - It is actually changing the weight matrix and the bias vector in each layer.
                  So if, y=wX+b in a linear layer, it is updating w & b such that it minimizes the loss function.
"""

"""
DC (2021-07-24) - So they are problem agnostic, any optimizer will work. Some just perform better than others depending
    on how difficult the optimization is.
"""

BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 10
IMAGE_DIM = 28

"""
DC (2021-07-24) - Lets go through this in our next meeting. It makes a lot more sense than what it looks,
                  see some comments inline below
DC (2021-07-24) - For pytorch datasets you need to write a __getitem__ method that accepts an index to tell the pytorch
                  dataloader how to index into your data. Similarly you need to write a __len__ method so that
                  the dataloader knows how big the dataset is and when to stop. They are required methods that get
                  called internally by dataloader, which is very powerful because now we can write arbitrary
                  dataloaders using plain python
"""


class ImageDataset(Dataset):
    def __init__(self, dirname):
        super(Dataset).__init__()
        self.dirname = dirname  # Accept a directory to read images from
        self.images = os.listdir(self.dirname)  # Get list of all files in above directory

    def __getitem__(self, index):
        image_name = self.images[index]  # Select image from image list in index
        image = read_image(os.path.join(self.dirname, image_name))  # Read image from disk into matrix
        image = resize(image, [IMAGE_DIM, IMAGE_DIM]) # Resize image to fit in memory during testing
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
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        """
        DC (2021-07-24) - This is correct, we know it will perform poorly
        DC (2021-07-24) - The last layer in the network should have a single node for binary classification
        """
        FLATTENED_DIM = (IMAGE_DIM ** 2) * 3
        self.layer_1 = nn.Linear(in_features=FLATTENED_DIM, out_features=FLATTENED_DIM)
        self.layer_2 = nn.Linear(in_features=FLATTENED_DIM, out_features=FLATTENED_DIM)
        self.layer_3 = nn.Linear(in_features=FLATTENED_DIM, out_features=FLATTENED_DIM)
        self.layer_4 = nn.Linear(in_features=FLATTENED_DIM, out_features=FLATTENED_DIM)
        self.layer_5 = nn.Linear(in_features=FLATTENED_DIM, out_features=int(FLATTENED_DIM * 0.5))
        self.layer_6 = nn.Linear(in_features=int(FLATTENED_DIM * 0.5), out_features=1)

    def forward(self, x):
        """
        DC (2021-07-29) - With the multiple layers now, you need a nonlinear activation function after each layer.
        """
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        x = F.relu(x)
        x = self.layer_4(x)
        x = F.relu(x)
        x = self.layer_5(x)
        x = F.relu(x)
        x = self.layer_6(x)

        """
        DC (2021-07-24) - Because we only have one hidden layer in the network, we do not need the relu as the sigmoid
                          will be the activation function.
        DC (2021-07-24) - This will change when we add more layers.
        """
        # x = F.relu(x)
        """
        DC (2021-07-24) - softmax is for multiclass classification
        DC (2021-07-24) - for binary classification last layer actication should be a sigmoid
        """
        # output = F.log_softmax(x, dim=1)
        output = F.sigmoid(x)  # signmoid is best for binary classification
        return output

def train(model, optimizer):
    for epoch in range(EPOCHS):
        for batch_number, (image, label) in enumerate(train_dataloader):
            image = image.view(-1, IMAGE_DIM * IMAGE_DIM * 3).to(torch.float32)
            """
            DC (2021-07-24) - Because image is cast to float32, need to cast label to float32 as well for the loss
                              calculation.
            CW (2021-07-25) - Why do we have to cast the image into a float? Doesn't the data-loader do that anyways?
            DC (2021-07-26) - They have to be floats for the graident calculations. The data-loader doesn't currently
                              do that but it can totally be written to do it there instead. In this case, the two
                              (image and label) just had to be cast to the same type.

                              Research is showing that training NN's can also work at lower floating point precision
                              without losing too much in accuracy. So to train the networks faster you will see people
                              casting down to f32, f16 and sometimes even f8.
            """
            label = label.to(torch.float32)
            prediction = model(image)
            """
            DC (2021-07-24) - In below loss there was a syntax error (missing bracket)
            """
            loss = F.binary_cross_entropy(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step

            if batch_number % 20 == 0:
                print(f"Epoch: {epoch} \t | Batch: {batch_number} \t | Loss: {loss}")

        val_loss = []
        for image, label in validation_dataloader:
            image = image.view(-1, IMAGE_DIM * IMAGE_DIM * 3).to(torch.float32)
            label = label.to(torch.float32)
            prediction = model(image)
            val_loss.append(F.binary_cross_entropy(prediction, label))
        print(f"Validation for epoch: {epoch} \t | Avg. Loss: {sum(val_loss)/len(val_loss)}")

if __name__ == "__main__":
    model = BinaryClassifier()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, optimizer)
