import torch
from torch.cuda.amp import autocast

from horch.train.learner import Learner, convert_tensor, backward, optimizer_step

from hinas.models.ppnas.search import Network


class PPNASLearner(Learner):

    def __init__(self, model: Network, criterion, optimizer_arch, optimizer_model, lr_scheduler,
                 grad_clip_norm=5, **kwargs):
        self.train_arch = True
        super().__init__(model, criterion, (optimizer_arch, optimizer_model),
                         lr_scheduler, grad_clip_norm=grad_clip_norm, **kwargs)

    def train_batch(self, batch):
        state = self._state['train']
        model = self.model
        optimizer_arch, optimizer_model = self.optimizers
        lr_scheduler = self.lr_schedulers[0]

        model.train()
        input, target = convert_tensor(self, batch)

        lr_scheduler.step(state['epoch'] + (state['step'] / state['steps']))
        optimizer_model.zero_grad(True)
        optimizer_arch.zero_grad(True)
        with autocast(enabled=self.fp16):
            logits = model(input)
            loss = self.criterion(logits, target)
        backward(self, loss)
        optimizer_step(self, optimizer_model, model.model_parameters())
        optimizer_step(self, optimizer_arch, model.arch_parameters())

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
        input, target = convert_tensor(self, batch)
        with autocast(enabled=self.fp16):
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
        input, target = convert_tensor(self, batch)
        with autocast(enabled=self.fp16):
            with torch.no_grad():
                output = model(input)

        state.update({
            "batch_size": input.size(0),
            "y_pred": output,
        })
