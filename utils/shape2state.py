import sys
import torch

model_path = sys.argv[1]
arch = sys.argv[2]
dest_path = sys.argv[3]

model = torch.load(model_path)
state = {
        'arch': arch,
        'state_dict': model.state_dict()
    }
torch.save(state, dest_path)