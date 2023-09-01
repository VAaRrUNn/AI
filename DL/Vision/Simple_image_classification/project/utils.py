import torch
import yaml
from tqdm.auto import tqdm
import torchvision.transforms as transforms
import os
print(os.getcwd())

# Configuration file
config_path = r"config/default_config.yaml"

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)


# Hyper parameters
BATCH_SIZE = config["training"]["batch_size"]
RESIZE_WIDTH = config["pre_processing"]["resize_width"]
RESIZE_HEIGHT = config["pre_processing"]["resize_height"]
SAVE_DIR = config["training"]["save_dir"]
MEAN = config["pre_processing"]["mean"]
STD = config["pre_processing"]["std"]

to_pil = transforms.Compose([
    transforms.ToPILImage(),
])

to_tensor = transforms.Compose([
    transforms.ToTensor()
])

train_transform = transforms.Compose([
    transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
    transforms.Normalize(mean=MEAN, std=STD),
])


def c_transform(image, transform=None):
    image = to_pil(image)
    image = image.convert('L')
    image = to_tensor(image)

    if transform:
        image = transform(image)

    return image


# resize = transforms.Compose([
#     transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH))
# ])


def normalization_values(dataset):
    """
    Except images of shape (C, H, W)
    Converts the images into GrayScale.
    Then calculates mean and std along channels.
    Returns the mean and std across each channel. of shape (1)
    """
    running_sum, running_sum_of_squares = 0, 0
    num_samples = len(dataset)

    for i in tqdm(range(num_samples)):
        image, _ = dataset[i]

        # stats
        running_sum += image.sum(dim=(1, 2))
        running_sum_of_squares += (image ** 2).sum(dim=(1, 2))

    mean = running_sum / (num_samples * image.size(1) * image.size(2))
    std = torch.sqrt((running_sum_of_squares / (num_samples *
                     RESIZE_HEIGHT * RESIZE_WIDTH - mean**2)))
    return mean, std


def accuracy_fn(y_true, y_pred):
    """
    Calculates accuracy given the predictions and the labels.
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = ((correct)/len(y_pred)) * 100
    return acc


def save_weights(*,
                 model):
    """
    Save the model weights in the directory mentioned in the config file.
    """
    # from model import model
    torch.save(model.state_dict(), SAVE_DIR)


# ----------------- TESTING -----------------------
test_transform = transforms.Compose([
    transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
    transforms.Normalize(mean=MEAN, std=STD),
])
