import torch

from flame.engine import Phase, BaseContext, context, context_field


@context
class Context(BaseContext):
    net: torch.nn.Module = context_field(default=None)
    optimizer: torch.optim.Optimizer = context_field(default=None)
    criterion: torch.nn.Module = context_field(default=None)
    scheduler: torch.optim.lr_scheduler._LRScheduler = context_field(default=None)

    train_phase: Phase = context_field(default=None)
    validation_phase: Phase = context_field(default=None)

    loss: torch.Tensor = context_field(default=None)
    datas: object = context_field(default=None)
    targets: object = context_field(default=None)
    outputs: object = context_field(default=None)

    def state_dict(self):
        pass
