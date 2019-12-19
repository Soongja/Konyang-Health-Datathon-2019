import torch.optim.lr_scheduler as lr_scheduler
from .warmup_scheduler.scheduler import GradualWarmupScheduler


def multi_step(optimizer, last_epoch, milestones=[500, 5000], gamma=0.1, **_):
    if isinstance(milestones, str):
        milestones = eval(milestones)
    return lr_scheduler.MultiStepLR(optimizer, milestones=milestones,
                                     gamma=gamma, last_epoch=last_epoch)


def cosine(optimizer, last_epoch, T_max=50, eta_min=0.00001, **_):
    print('cosine annealing, T_max: {}, eta_min: {}, last_epoch: {}'.format(T_max, eta_min, last_epoch))
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min,
                                           last_epoch=last_epoch)


def one_cycle_lr(optimizer, last_epoch, max_lr, pct_start, epochs, steps_per_epoch, anneal_strategy='cos', **_):
    return lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch, pct_start=pct_start,
                                   anneal_strategy=anneal_strategy, last_epoch=last_epoch)


def reduce_lr_on_plateau(optimizer, last_epoch, mode='max', factor=0.1, patience=10,
                         threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, **_):
    return lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience,
                                           threshold=threshold, threshold_mode=threshold_mode,
                                           cooldown=cooldown, min_lr=min_lr)


def none(optimizer, last_epoch, **_):
    return lr_scheduler.StepLR(optimizer, step_size=10000000, last_epoch=last_epoch)


def get_scheduler(config, optimizer, last_epoch):
    print('scheduler name:', config.SCHEDULER.NAME, 'warm up:', config.SCHEDULER.WARMUP)

    f = globals().get(config.SCHEDULER.NAME)

    if config.SCHEDULER.PARAMS is None:
        scheduler = f(optimizer, last_epoch)
    else:
        scheduler = f(optimizer, last_epoch, **config.SCHEDULER.PARAMS)

    if config.SCHEDULER.WARMUP:
        scheduler = GradualWarmupScheduler(optimizer, after_scheduler=scheduler, **config.SCHEDULER.WARMUP_PARAMS)

    return scheduler
