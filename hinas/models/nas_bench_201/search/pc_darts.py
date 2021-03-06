from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from hinas.models.darts.search.pc_darts import beta_softmax, channel_shuffle
from hinas.models.operations import OPS
from hinas.models.primitives import PRIMITIVES_nas_bench_201
import hinas.models.nas_bench_201.search.darts as darts


class MixedOp(nn.Module):

    def __init__(self, C, k):
        super().__init__()
        self.k = k
        self._channels = C // k

        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES_nas_bench_201:
            op = OPS[primitive](C, 1)
            self._ops.append(op)

    def forward(self, x, weights):
        x1 = x[:, :self._channels, :, :]
        x2 = x[:, self._channels:, :, :]
        x1 = sum(w * op(x1) for w, op in zip(weights, self._ops))
        x = torch.cat([x1, x2], dim=1)
        x = channel_shuffle(x, self.k)
        return x


class Cell(darts.Cell):

    def __init__(self, steps, C_prev, C, k):
        super().__init__(steps, C_prev, C, partial(MixedOp, k=k))


class Network(darts.Network):

    def __init__(self, C, layers, steps=3, stem_multiplier=3, k=4, num_classes=10):
        self._k = k
        super().__init__(C, layers, steps, stem_multiplier, num_classes,
                         partial(Cell, k=k))

    def _initialize_alphas(self):
        k = sum(1 + i for i in range(self._steps))
        num_ops = len(PRIMITIVES_nas_bench_201)

        self.alphas = nn.Parameter(1e-3 * torch.randn(k, num_ops), requires_grad=True)
        self.betas = nn.Parameter(1e-3 * torch.randn(k), requires_grad=True)

    def forward(self, x):
        s = self.stem(x)
        alphas = F.softmax(self.alphas_reduce, dim=1)
        betas = beta_softmax(self.betas_normal, self._steps)

        for cell in self.cells:
            s = cell(s, alphas, betas)
        x = self.avg_pool(s)
        logits = self.classifier(x)
        return logits

    def arch_parameters(self):
        return [self.alphas, self.betas]

    def model_parameters(self):
        ids = set(id(p) for p in self.arch_parameters())
        for p in self.parameters():
            if id(p) not in ids:
                yield p

    def genotype(self):
        alphas = F.softmax(self.alphas.detach().cpu(), dim=1).numpy()
        betas = beta_softmax(self.betas, self._steps).numpy()
        alphas = alphas * betas[:, None]

        gene = darts.parse_weights(alphas, self._steps)
        s = "+".join([f"|{'|'.join(f'{op}~{i}' for op, i in ops)}|" for ops in gene])
        return s
