from project.setup import train
from project.utils import save_weights
from project.model import model
# import matplotlib.pyplot as plt

print("Finally importing done...")
if __name__ == "__main__":
    loss, acc = train()

    # fig, ax = plt.subplots(1, 2)
    # plt.figure(figsize=(5, 15))

    # ax[0].plot(range(len(loss)), loss,)
    # ax[1].plot(range(len(acc)), acc)

    save_weights(model=model)
