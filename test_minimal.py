from minimal import GAN
import torch

model = GAN(21, 2, int(2.5 * 512), 6, 120, 210)



print('---' * 20)

# print(model.critic)

z = torch.randn(1, 210)

print(z.shape)
y = torch.tensor([1], dtype=torch.int32)

model(z, y)