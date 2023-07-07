from train import *
from model import *

N_EPOCHS = 10

n_epochs = 1
train(n_epochs=N_EPOCHS,
      dataloader=dataloader,
      loss_fn=loss_fn,
      gen_a=gen_a,
      gen_b=gen_b,
      disc_a=disc_a,
      disc_b=disc_b,
      gen_opt_a=gen_opt_a,
      gen_opt_b=gen_opt_b,
      disc_opt_a=disc_opt_a,
      disc_opt_b=disc_opt_b,
      device=device,
      )

def count_params(model):
    return sum(map(lambda p: p.data.numel(), model.parameters()))

total_param = 0
for model in [gen_a, gen_b, disc_a, disc_b]:
    total_param += count_params(model=model)

print(total_param)
