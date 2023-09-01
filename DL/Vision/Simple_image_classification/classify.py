from project.utils import test_transform, to_tensor
from project.model import model
import torch
from PIL import Image
import argparse


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Make it dynamic too.


# classifying
# cat = 0, dog = 1

@torch.no_grad()
def predict(model,
            image,
            device,
            threshold=0.5):
    model.eval()

    image = Image.open(image)
    image = image.convert('L')

    image = to_tensor(image).unsqueeze(1)
    image = test_transform(image)

    model, image = model.to(device), image.to(device)
    with torch.inference_mode():
        logits = model(image)
        pred = torch.sigmoid(logits)

    if pred >= threshold:
        return 1
    return 0


def classify(image):
    pred = predict(model=model,
                   image=image,
                   device=DEVICE)
    ans = ""
    if pred == 1:
        ans = "dog"
    else:
        ans = "cat"

    print(f"The given image is of a {ans}")


if __name__ == "__main__":

    # argparse
    parser = argparse.ArgumentParser(description="Classify image (dog or cat)")
    parser.add_argument("-i", "--image", help="Image path")
    args = parser.parse_args()

    image = args.image
    classify(image=image)
