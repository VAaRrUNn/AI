import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# hard coded rn
images_dir = r"D:\material\Machine_Deep\github_repos\DL\resources\CNN_architectures\semantic_drone_dataset\original_images"
labels_dir = r"D:\material\Machine_Deep\github_repos\DL\resources\CNN_architectures\RGB_color_image_masks"



class segmentationDataset(Dataset):
    def __init__(self, images_dir,
                 labels_dir):

        self.images = os.listdir(images_dir)
        self.labels = os.listdir(labels_dir)

        self.images = [os.path.join(images_dir, image)
                       for image in self.images]
        self.labels = [os.path.join(labels_dir, label)
                       for label in self.labels]

        self.resizing = transforms.Compose([
            transforms.Resize((572, 572)),
        ])

        self.toTensor = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]
        img, label = Image.open(img), Image.open(label)
        img = self.resizing(self.toTensor(img))
        label = self.resizing(self.toTensor(label))
        return (img, label)


dataset = segmentationDataset(images_dir, labels_dir)

dataloader = DataLoader(dataset,
                        batch_size=2,
                        shuffle=True)