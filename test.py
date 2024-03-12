import torch
from torch.distributions import Bernoulli

# 각 요소가 1이 될 확률을 나타내는 텐서 생성
probs = torch.rand(30, 30)  # 0과 1 사이의 무작위 값

# Bernoulli 분포 정의
dist = Bernoulli(probs)

# 샘플링
sample = dist.sample()

print(sample.shape)  # (30, 30)