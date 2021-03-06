import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose, RandomCrop, RandomHorizontalFlip

from hhutil.io import parse_python_config

from horch.datasets import train_test_split
from horch.defaults import set_defaults
from horch.optim.lr_scheduler import CosineLR
from horch.train.metrics import TrainLoss, Loss
from horch.train.cls.metrics import Accuracy
from horch.train import manual_seed

from hinas.models.primitives import set_primitives
from hinas.train.darts import DARTSLearner

cfg = parse_python_config("config.py")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
manual_seed(cfg.seed)

train_transform = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize([0.491, 0.482, 0.447], [0.247, 0.243, 0.262]),
])

valid_transform = Compose([
    ToTensor(),
    Normalize([0.491, 0.482, 0.447], [0.247, 0.243, 0.262]),
])

root = 'datasets/CIFAR10'
ds = CIFAR10(root, train=True, download=True)

ds_train, ds_search = train_test_split(
    ds, test_ratio=0.5, shuffle=True, random_state=cfg.seed,
    transform=train_transform, test_transform=train_transform)
ds_val = train_test_split(
    ds, test_ratio=0.5, shuffle=True, random_state=cfg.seed,
    test_transform=valid_transform)[1]

train_loader = DataLoader(ds_train, batch_size=cfg.train_batch_size, pin_memory=True, shuffle=True, num_workers=2)
search_loader = DataLoader(ds_search, batch_size=cfg.search_batch_size, pin_memory=True, shuffle=True, num_workers=2)
val_loader = DataLoader(ds_val, batch_size=cfg.val_batch_size, pin_memory=True, shuffle=False, num_workers=2)

set_defaults({
    'relu': {
        'inplace': False,
    },
    'bn': {
        'affine': False,
    }
})
set_primitives(cfg.primitives)
model = cfg.network_fn()
criterion = nn.CrossEntropyLoss()

epochs = cfg.epochs
optimizer_arch = Adam(model.arch_parameters(), lr=cfg.arch_lr, betas=(0.5, 0.999), weight_decay=1e-3)
optimizer_model = SGD(model.model_parameters(), cfg.model_lr, momentum=0.9, weight_decay=getattr(cfg, "model_wd", 3e-4))
lr_scheduler = CosineLR(optimizer_model, epochs, min_lr=getattr(cfg, "model_min_lr", 0))

train_metrics = {
    "loss": TrainLoss(),
    "acc": Accuracy(),
}

eval_metrics = {
    "loss": Loss(criterion),
    "acc": Accuracy(),
}

trainer = DARTSLearner(model, criterion, optimizer_arch, optimizer_model, lr_scheduler,
                       train_metrics=train_metrics, eval_metrics=eval_metrics,
                       search_loader=search_loader, grad_clip_norm=cfg.grad_clip_norm, work_dir=cfg.work_dir)

trainer.fit(search_loader, epochs, val_loader, val_freq=getattr(cfg, "val_freq", 5), callbacks=cfg.callbacks)