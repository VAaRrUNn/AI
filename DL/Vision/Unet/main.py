from train import *

N_EPOCHS = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet().to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

train(n_epochs=N_EPOCHS,
      dataloader=dataloader,
      optimizer=optimizer,
      loss_fn=loss_fn,
      model=model,
      device=device)
