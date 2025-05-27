import torch
from pytorch_ops import soft_rank, soft_sort

REG_STRENGTH = 0.1
s1 = torch.rand(100).reshape(1, 100)
print(f's1 = {s1}')

s1_sorted = -soft_sort(-s1, regularization_strength=REG_STRENGTH,device='cuda:0')
print(f's1_sorted = {s1_sorted}')

s1_rank = soft_rank(-s1, regularization_strength=REG_STRENGTH,device='cuda:0')-1
print(f's1_rank = {s1_rank}')

print(f's1 torch sort = {torch.sort(-s1, dim =1)}')

