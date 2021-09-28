from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, \
    CosineAnnealingLR, CosineAnnealingWarmRestarts, CyclicLR, OneCycleLR


def get_schedulers_dict(cfg, samples_per_epoch):
    base_lr = cfg.args.lr
    gamma = cfg.train.gamma
    epochs = cfg.train.epochs

    schedulers = {
        "StepLR_1": [StepLR, {'step_size': 1, 'gamma': gamma}, 'step'],
        "StepLR_2": [StepLR, {'step_size': 2, 'gamma': gamma}, 'step'],
        "StepLR_3": [StepLR, {'step_size': 3, 'gamma': gamma}, 'step'],
        "ReduceLROnPlateau_2": [ReduceLROnPlateau,
                                {'mode': 'min', 'factor': gamma, 'patience': 2},
                                'plateau'],
        "ReduceLROnPlateau_3": [ReduceLROnPlateau,
                                {'mode': 'min', 'factor': gamma, 'patience': 3},
                                'plateau'],
        "CosineAnnealingLR": [CosineAnnealingLR,
                              {'T_max': 10, 'eta_min': 0},
                              'cosine'],
        "CosineAnnealingWarmRestarts": [CosineAnnealingWarmRestarts,
                                        {'T_0': 10, 'T_mult': 1, 'eta_min': 0},
                                        'cosine'],
        "CyclicLR_triangular": [CyclicLR,
                                {'base_lr': base_lr,
                                 'max_lr': 0.1,
                                 'mode': 'triangular'},
                                'cyclic'],
        "CyclicLR_triangular2": [CyclicLR,
                                 {'base_lr': base_lr,
                                  'max_lr': 0.1,
                                  'mode': 'triangular2'},
                                 'cyclic'],
        "CyclicLR_exp_range": [CyclicLR,
                               {'base_lr': base_lr,
                                'max_lr': 0.1,
                                'mode': 'exp_range'},
                               'cyclic'],
        "OneCycleLR_cos": [OneCycleLR,
                           {'max_lr': 0.1,
                            'steps_per_epoch': samples_per_epoch,
                            'epochs': epochs,
                            'anneal_strategy': 'cos'},
                           'cyclic'],
        "OneCycleLR_linear": [OneCycleLR,
                              {'max_lr': 0.1,
                               'steps_per_epoch': samples_per_epoch,
                               'epochs': epochs,
                               'anneal_strategy':  'linear'},
                              'cyclic']
    }
    return schedulers
