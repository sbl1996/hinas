import torch
import torch.nn as nn

from horch.common import convert_tensor
from horch.train.learner import Learner

from hinas.models.darts.search.darts import Network

def requires_grad(network: Network, arch: bool, model: bool):
    for p in network.arch_parameters():
        p.requires_grad_(arch)
    for p in network.model_parameters():
        p.requires_grad_(model)


class DARTSLearner(Learner):

    def __init__(self, model: Network, criterion, optimizer_arch, optimizer_model, lr_scheduler,
                 grad_clip_norm=5, search_loader=None, **kwargs):
        self.train_arch = True
        self.search_loader = search_loader
        self.search_loader_iter = None
        super().__init__(model, criterion, (optimizer_arch, optimizer_model),
                         lr_scheduler, grad_clip_norm=grad_clip_norm, **kwargs)

    def get_search_batch(self):
        if self.search_loader_iter is None:
            self.search_loader_iter = iter(self.search_loader)
        try:
            return next(self.search_loader_iter)
        except StopIteration:
            self.search_loader_iter = iter(self.search_loader)
            return self.get_search_batch()

    def train_batch(self, batch):
        state = self._state['train']
        model = self.model
        optimizer_arch, optimizer_model = self.optimizers
        lr_scheduler = self.lr_schedulers[0]

        model.train()
        input, target = convert_tensor(batch, self.device)

        if self.train_arch:
            input_search, target_search = convert_tensor(self.get_search_batch(), self.device)
            requires_grad(model, arch=True, model=False)
            optimizer_arch.zero_grad()
            logits_search = model(input_search)
            loss_search = self.criterion(logits_search, target_search)
            loss_search.backward()
            optimizer_arch.step()

        requires_grad(model, arch=False, model=True)
        lr_scheduler.step(state['epoch'] + (state['step'] / state['steps']))
        optimizer_model.zero_grad()
        logits = model(input)
        loss = self.criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.model_parameters(), self.grad_clip_norm)
        optimizer_model.step()

        state.update({
            "loss": loss.item(),
            "batch_size": input.size(0),
            "y_true": target,
            "y_pred": logits.detach(),
        })

    def eval_batch(self, batch):
        state = self._state['eval']
        model = self.model

        model.eval()
        input, target = convert_tensor(batch, self.device)
        with torch.no_grad():
            output = model(input)

        state.update({
            "batch_size": input.size(0),
            "y_true": target,
            "y_pred": output,
        })

    def test_batch(self, batch):
        state = self._state['test']
        model = self.model

        model.eval()
        input = convert_tensor(batch, self.device)
        with torch.no_grad():
            output = model(input)

        state.update({
            "batch_size": input.size(0),
            "y_pred": output,
        })
