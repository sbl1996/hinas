import numpy as np

import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose, RandomCrop, RandomHorizontalFlip

from horch.datasets import train_test_split, CombineDataset
from horch.defaults import set_defaults

from horch.optim.lr_scheduler import CosineAnnealingLR
from horch.train import manual_seed
from horch.train.cls.metrics import Accuracy
from horch.train.callbacks import Callback
from horch.train.metrics import TrainLoss, Loss

from hinas.models.nas_bench_201.api import SimpleNASBench201
from hinas.models.nas_bench_201.search.gdas import Network
from hinas.train.darts.callbacks import TauSchedule
from hinas.train.darts import DARTSLearner

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
ds = train_test_split(ds, test_ratio=0.005, shuffle=True)[1]
ds_train, ds_search = train_test_split(
    ds, test_ratio=0.5, shuffle=True, random_state=42,
    transform=train_transform, test_transform=train_transform)
ds_val = train_test_split(
    ds, test_ratio=0.5, shuffle=True, random_state=42,
    test_transform=valid_transform)[1]

train_loader = DataLoader(ds_train, batch_size=32, pin_memory=True, shuffle=True, num_workers=2)
search_loader = DataLoader(ds_search, batch_size=32, pin_memory=True, shuffle=True, num_workers=2)
val_loader = DataLoader(ds_val, batch_size=32, pin_memory=True, shuffle=False, num_workers=2)

api = SimpleNASBench201("/Users/hrvvi/Code/study/pytorch/datasets/NAS-Bench-201-v1_1-096897-simple.pth")

set_defaults({
    'relu': {
        'inplace': False,
    },
    'bn': {
        'affine': False,
    }
})
model = Network(4, 8)
criterion = nn.CrossEntropyLoss()

epochs = 250
optimizer_model = SGD(model.model_parameters(), 0.025, momentum=0.9, weight_decay=3e-4, nesterov=True)
optimizer_arch = Adam(model.arch_parameters(), lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-3)
lr_scheduler = CosineAnnealingLR(optimizer_model, epochs=epochs, min_lr=1e-5)

train_metrics = {
    "loss": TrainLoss(),
    "acc": Accuracy(),
}

eval_metrics = {
    "loss": Loss(criterion),
    "acc": Accuracy(),
}


learner = DARTSLearner(
    model, criterion, optimizer_arch, optimizer_model, lr_scheduler,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    search_loader=search_loader, work_dir='models/DARTS', fp16=False)


class EvalGenotype(Callback):

    def __init__(self, api, dataset='cifar10'):
        super().__init__()
        self.api = api
        self.dataset = dataset

    def after_epoch(self, state):
        g = self.learner.model.genotype()
        print(g)
        print("*************************")
        acc = np.mean(self.api.query_eval_acc(g, self.dataset))
        rank = self.api.query_eval_acc_rank(g, self.dataset)
        print("**%d**%.2f**" % (rank, acc))
        print("*************************")


learner.fit(train_loader, epochs, val_loader,
            callbacks=[TauSchedule(tau_max=10, tau_min=0.1), EvalGenotype(api)])
