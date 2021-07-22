import os
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 1


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

if __name__ == "__main__":
    for batch_number, (image, label) in enumerate(train_dataloader):
        print(image, label)
        break
