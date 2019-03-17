import torch
import flame

import torch.utils.data
from torchvision import datasets
import torchvision.transforms

from core import lenet, Context
from flame.engine import Phase

ctx = Context()

ctx.max_epoch = flame.hocon.get_int("strategy.epoch")

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=flame.hocon.get_list("dataset.mean"),
        std=flame.hocon.get_list("dataset.std"),
    ),
])
train_dataset = datasets.MNIST(
    root=flame.hocon.get_string("dataset.root"),
    train=True,
    transform=transform,
    download=True,
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=flame.hocon.get_int("strategy.train.batch_size"),
    shuffle=True,
    num_workers=flame.hocon.get_int("machine.n_loader_processes"),
)
ctx.train_phase = Phase("training", train_loader)

val_dataset = datasets.MNIST(
    root=flame.hocon.get_string("dataset.root"),
    train=False,
    transform=transform,
    download=True,
)
val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=flame.hocon.get_int("strategy.val.batch_size"),
    num_workers=flame.hocon.get_int("machine.n_loader_processes"),
)
ctx.validation_phase = Phase("validation", val_loader)

ctx.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

ctx.net = lenet(num_classes=flame.hocon.get_int("dataset.n_classes"))

ctx.net = ctx.net.to(ctx.device)

if flame.hocon.get_bool("machine.parallel"):
    ctx.net = torch.nn.DataParallel(ctx.net)

ctx.criterion = torch.nn.CrossEntropyLoss()

ctx.optimizer = torch.optim.SGD(
    params=ctx.net.parameters(),
    lr=flame.hocon.get_float("optimizer.learning_rate"),
    momentum=flame.hocon.get_float("optimizer.momentum"),
    dampening=flame.hocon.get_float("optimizer.dampening"),
    weight_decay=flame.hocon.get_float("optimizer.weight_decay"),
    nesterov=flame.hocon.get_bool("optimizer.nesterov")
)

ctx.scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer=ctx.optimizer,
    milestones=flame.hocon.get_list("scheduler.milestones"),
    gamma=flame.hocon.get_float("scheduler.gamma")
)
