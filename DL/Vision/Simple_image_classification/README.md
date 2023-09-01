This is a simple project on Cat vs Dog classification.
It makes use of [cats_vs_dogs](https://huggingface.co/datasets/cats_vs_dogs) dataset available on [HuggingFace](https://huggingface.co/).

It is fine-tuned on the above dataset using [vgg19](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html) model available on [Pytorch](https://pytorch.org/).


<div align="center">
    <h1>Quick Start</h1>
    <a href="https://colab.research.google.com/drive/1RX8j65oo5yLxjj7MB3PHpYuOV9ANVXqC?usp=sharing"> <img src="https://colab.research.google.com/assets/colab-badge.svg"> </a>
</div>

<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RX8j65oo5yLxjj7MB3PHpYuOV9ANVXqC?usp=sharing) -->





# Installation

### **Using pip**:

```
pip install datasets tqdm torch numpy torchvision yaml
```

### **Using conda**

```
conda install datasets tqdm torch numpy torchvision yaml
```


# Training:

```
python .\train.py
```


# Inference

```
python .\classify.py -i image_path
```

OR

```
python .\classify.py --image image_path
```