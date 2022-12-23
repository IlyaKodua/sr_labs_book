# The script is borrowed from the following repository: https://github.com/clovaai/voxceleb_trainer
# The script creates different schedulers


# Import of modules
import torch


def StepLRScheduler(optimizer, test_interval, lr_decay, **kwargs):
    # Function to create scheduler
    
    sche_fn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    lr_step = 'epoch'

    print('Initialised step LR scheduler.')

    return sche_fn, lr_step

def OneCycleLRScheduler(optimizer, pct_start, cycle_momentum, max_lr, div_factor, final_div_factor, total_steps, **kwargs):
    # Function to create scheduler
    
    sche_fn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    lr_step = 'iteration'

    print('Initialised OneCycle LR scheduler.')

    return sche_fn, lr_step