import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose, RandomCrop, RandomHorizontalFlip

from horch.datasets import train_test_split
from horch.defaults import set_defaults
from horch.train.metrics import TrainLoss, Loss
from horch.train.cls.metrics import Accuracy

from horch.train import manual_seed

from hinas.models.darts.search.pc_darts import Network
from hinas.train.darts import DARTSLearner, TrainArchSchedule

manual_seed(0)

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


root = '/Users/hrvvi/Code/study/pytorch/datasets/CIFAR10'
ds = CIFAR10(root, train=True)
ds = train_test_split(ds, test_ratio=0.003, shuffle=True)[1]
ds_train, ds_search = train_test_split(
    ds, test_ratio=0.5, shuffle=True, random_state=42,
    transform=train_transform, test_transform=train_transform)
ds_val = train_test_split(
    ds, test_ratio=0.5, shuffle=True, random_state=42,
    test_transform=valid_transform)[1]

train_loader = DataLoader(ds_train, batch_size=32, pin_memory=True, shuffle=True, num_workers=2)
search_loader = DataLoader(ds_search, batch_size=32, pin_memory=True, shuffle=True, num_workers=2)
val_loader = DataLoader(ds_val, batch_size=32, pin_memory=True, shuffle=False, num_workers=2)

set_defaults({
    'relu': {
        'inplace': False,
    },
    'bn': {
        'affine': False,
    }
})
model = Network(8, 8, steps=4, multiplier=4, stem_multiplier=1, k=4)
# model = Network(16, 8, k=4)
criterion = nn.CrossEntropyLoss()

optimizer_arch = Adam(model.arch_parameters(), lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-3)
optimizer_model = SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3e-4)
lr_scheduler = CosineAnnealingLR(optimizer_model, T_max=50, eta_min=0.001)

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
                       search_loader=search_loader, work_dir='models/DARTS')


trainer.fit(search_loader, 50, val_loader, val_freq=2,
            callbacks=[TrainArchSchedule(after_epochs=15)])

# trainer.resume()
#
# trainer.fit(train_loader, None, val_loader, eval_freq=2)