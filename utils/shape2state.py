import sys
import torch

model_path = sys.argv[1]
dest_path = sys.argv[2]
model = torch.load(model_path)
torch.save(model.state_dict(), dest_path)