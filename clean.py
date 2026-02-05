import torch 
torch.cuda.empty_cache()
print(torch.cuda.memory_summary(device=None, abbreviated=False))