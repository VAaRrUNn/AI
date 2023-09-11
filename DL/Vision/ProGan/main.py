from dataset import dataloader, dataset
from model import *
from utils import *
from train import *

N_EPOCHS = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = dataset
dataloader = dataloader


gen = gen.to(device)
disc = disc.to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=0.01)
disc_opt = torch.optim.Adam(disc.parameters(), lr=0.01)
loss_fn = nn.BCEWithLogitsLoss()

losses = train(n_epochs=N_EPOCHS,
               gen=gen,
               disc=disc,
               dataloader=dataloader,
               gen_opt=gen_opt,
               disc_opt=disc_opt,
               device=device,
               loss_fn=loss_fn,
               steps=NUM_BLOCKS)

gen_losses_se, disc_losses_se, gen_losses_e, disc_losses_e, gen_losses_b, disc_losses_b = losses
