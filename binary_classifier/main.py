import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam # only optimizer I could find for binary classification
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 1
LEARNING_RATE = 0.001
EPOCHS = 5

# ???
class ImageDataset(Dataset):
    def __init__(self, dirname):
        super(Dataset).__init__()
        self.dirname = dirname
        self.images = os.listdir(self.dirname)

    def __getitem__(self, index):
        image_name = self.images[index]
        image = read_image(os.path.join(self.dirname, image_name))
        if image_name.split("_")[0] == "photo":
            label = torch.tensor([0])
        else:
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
        self.layer_1 = nn.Linear(in_features=256*256*3, out_features=1) #???

    def forward(self, x):
        x = self.layer_1(x)
        x = F.relu(x)
        output = F.log_softmax(x, dim=1) #??? dim????
        return output

def train(model, optimizer):
    for epoch in range(EPOCHS):
        for batch_number, (image, label) in enumerate(train_dataloader):
            image = image.view(-1, 256*256*3).to(torch.float32)
            prediction = model(image)
            loss = F.binary_cross_entropy(prediction.to(torch.float32, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            if batch_number % 10 == 0:
                print(f"Epoch: {epoch} \t |Batch: {batch_number} \t | Loss: {loss}")

if __name__ == "__main__":
    model = BinaryClassifier()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, optimizer)
    # for batch_number, (image, label) in enumerate(train_dataloader):
    #     print(image.size()) # view???? -> turns tensor into different shape
    #     break


