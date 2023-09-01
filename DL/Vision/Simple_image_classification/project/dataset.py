from .utils import to_tensor, c_transform, BATCH_SIZE, train_transform
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class train_dataset(Dataset):
    def __init__(self,
                 *,
                 data,
                 c_transform=None,
                 transform=None):
        self.data = data
        self.transform = transform
        self.c_transform = c_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self,
                    index):
        pos = self.data[index]
        image, label = pos["image"], pos["labels"]
        if self.c_transform:
            image = to_tensor(image)
            image = self.c_transform(image, self.transform)
        elif self.transform:
            image = to_tensor(image)
            image = self.transform(image)

        return image, label


print("ready for training....")
train_set = load_dataset("cats_vs_dogs", split="train",
                         ignore_verifications=True)
print("done with downloading...")
dataset = train_dataset(data=train_set, c_transform=c_transform)


# No need to recalculate it
# mean, std = normalization_values(dataset)


dataset = train_dataset(data=train_set,
                        c_transform=c_transform,
                        transform=train_transform)


train_dataloader = DataLoader(dataset=dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
