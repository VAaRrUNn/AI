import torch.nn as nn
import torch
import torchvision.models as models
from .utils import SAVE_DIR


class ModifiedVGG(nn.Module):
    def __init__(self):
        super(ModifiedVGG, self).__init__()

        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.vgg = nn.Sequential(
            vgg.features,
            vgg.avgpool,
        )
        self.fix_input = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=3,
                      kernel_size=3,
                      stride=2,
                      padding=0)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256,
                      kernel_size=3, stride=3),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(inplace=False),
            nn.Linear(in_features=256, out_features=32, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(inplace=False),
            nn.Linear(in_features=32, out_features=1, bias=True)
        )

    def forward(self, images):
        out = self.fix_input(images)
        out = self.vgg(out)
        out = self.classifier(out)
        return out


model = ModifiedVGG()

# Load pre-trained weights
model.load_state_dict(torch.load(SAVE_DIR))

# Training only the last 2 layers.
for name, params in model.named_parameters():
    if name.startswith("vgg"):
        params.requires_grad = False
