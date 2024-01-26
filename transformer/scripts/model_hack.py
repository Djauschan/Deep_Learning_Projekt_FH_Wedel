import torch

model = torch.load("data\\output\\models\\TransformerModel_v1.pt")

m = torch.jit.trace(model, torch.rand(1, 96, 28).to(torch.device("cuda")))
torch.jit.save(m, "data\\output\\models\\TransformerModel_v1_jit.tjm")
