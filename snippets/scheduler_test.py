import torch
import torch.optim.lr_scheduler as lr_scheduler
from warmup_scheduler import GradualWarmupScheduler
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


v = torch.zeros(10)
lr = 0.001
optim = torch.optim.SGD([v], lr=lr)
optim.param_groups[0]['initial_lr'] = lr


last_epoch = -1
scheduler = lr_scheduler.MultiStepLR(optim, milestones=[4], gamma=0.1, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10, eta_min=0.00001, last_epoch=-1)
# scheduler = lr_scheduler.OneCycleLR(optim, max_lr=0.001, total_steps=6000, pct_start=0.033, anneal_strategy='cos', last_epoch=last_epoch)

warmup = True
if warmup:
    scheduler = GradualWarmupScheduler(optim, multiplier=5, total_epoch=5, after_scheduler=scheduler)

# if last_epoch != -1:
#     scheduler.step()



lrs = []
for epoch in range(last_epoch+1, 30):
    print(epoch, optim.param_groups[0]['lr'])
    lrs.append(optim.param_groups[0]['lr'])

    scheduler.step()

plt.plot(lrs)
plt.show()