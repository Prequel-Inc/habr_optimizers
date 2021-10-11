import torch
import torch_optimizer as optim

from common.madgrad.madgrad import MADGRAD

optimizers = {
    "MADGRAD": MADGRAD,
    "Adam": torch.optim.Adam,
    "Adadelta": torch.optim.Adadelta,
    "AdamW": torch.optim.AdamW,
    "Adamax": torch.optim.Adamax,
    "Adagrad": torch.optim.Adagrad,
    "ASGD": torch.optim.ASGD,
    "RMSprop": torch.optim.RMSprop,
    "Rprop": torch.optim.Rprop,
    "SGD": torch.optim.SGD,
    "Yogi": optim.Yogi,
    "RAdam": optim.RAdam,
    "Lookahead": optim.Lookahead,
    "NovoGrad": optim.NovoGrad,
    "Ranger": optim.Ranger,
    "A2GradExp": optim.A2GradExp,
    "A2GradInc": optim.A2GradInc,
    "A2GradUni": optim.A2GradUni,
    "AccSGD": optim.AccSGD,
    "AdaBelief": optim.AdaBelief,
    "AdaBound": optim.AdaBound,
    "AdaMod": optim.AdaMod,
    "Adahessian": optim.Adahessian,
    "AdamP": optim.AdamP,
    "AggMo": optim.AggMo,
    "Apollo": optim.Apollo,
    "DiffGrad": optim.DiffGrad,
    "Lamb": optim.Lamb,
    "PID": optim.PID,
    "QHAdam": optim.QHAdam,
    "QHM": optim.QHM,
    "SGDP": optim.SGDP,
    "SGDW": optim.SGDW,
    "SWATS": optim.SWATS,
    "RangerQH": optim.RangerQH,
    "RangerVA": optim.RangerVA,
    # not used in experiments
    # "LBFGS": torch.optim.LBFGS,
    # "Adafactor": optim.Adafactor,
    # "Shampoo": optim.Shampoo
}

opts_without_momentum = ["Adadelta", "Adagrad", "ASGD",
                         "LBFGS", "Rprop", "A2GradExp",
                         "A2GradInc", "A2GradUni", "AccSGD",
                         "AggMo", "Apollo", "RangerVA"]
