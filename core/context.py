import torch

from flame.engine import Phase, BaseContext, context, field


@context
class Context(BaseContext):
    net: torch.nn.Module = field(default=None)
    optimizer: torch.optim.Optimizer = field(default=None)
    criterion: torch.nn.Module = field(default=None)
    scheduler: torch.optim.lr_scheduler._LRScheduler = field(default=None)

    train_phase: Phase = field(default=None)
    validation_phase: Phase = field(default=None)

    loss: torch.Tensor = field(default=None)
    datas: object = field(default=None)
    targets: object = field(default=None)
    outputs: object = field(default=None)
