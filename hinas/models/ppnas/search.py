import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.nn import GlobalAvgPool
from horch.models.layers import Conv2d, Norm, Act, Linear, Pool2d, Sequential, Identity

from hinas.models.ppnas.operations import OPS
from hinas.models.ppnas.primitives import get_primitives


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in get_primitives():
            op = OPS[primitive](C, stride)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class PPConv(nn.Module):

    def __init__(self, channels, splits):
        super().__init__()
        self.splits = splits
        C = channels // splits

        self.ops = nn.ModuleList()
        for i in range(self.splits):
            op = MixedOp(C, 1)
            self.ops.append(op)

    def forward(self, x, alphas, betas):
        states = list(torch.split(x, x.size(1) // self.splits, dim=1))
        offset = 0
        for i in range(self.splits):
            x = sum(alphas[offset + j] * h for j, h in enumerate(states))
            x = self.ops[i](x, betas[i])
            offset += len(states)
            states.append(x)

        return torch.cat(states[-self.splits:], dim=1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride, base_width, splits,
                 start_block=False, end_block=False, exclude_bn0=False):
        super().__init__()
        self.stride = stride

        out_channels = channels * self.expansion
        width = math.floor(out_channels // self.expansion * (base_width / 64)) * splits
        if not start_block and not exclude_bn0:
            self.bn0 = Norm(in_channels)
        if not start_block:
            self.act0 = Act()
        self.conv1 = Conv2d(in_channels, width, kernel_size=1)
        self.bn1 = Norm(width)
        self.act1 = Act()
        if stride == 1:
            self.conv2 = PPConv(width, splits=splits)
        else:
            self.conv2 = Conv2d(width, width, kernel_size=3, stride=2,
                                groups=splits)
        self.conv3 = Conv2d(width, out_channels, kernel_size=1)

        if start_block:
            self.bn3 = Norm(out_channels)

        if end_block:
            self.bn3 = Norm(out_channels)
            self.act3 = Act()

        if stride != 1 or in_channels != out_channels:
            shortcut = []
            if stride != 1:
                shortcut.append(Pool2d(2, 2, type='avg'))
            shortcut.append(
                Conv2d(in_channels, out_channels, kernel_size=1, norm='def'))
            self.shortcut = Sequential(shortcut)
        else:
            self.shortcut = Identity()
        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def forward(self, x, alphas, betas):
        identity = self.shortcut(x)
        if self.start_block:
            x = self.conv1(x)
        else:
            if not self.exclude_bn0:
                x = self.bn0(x)
            x = self.act0(x)
            x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.stride == 1:
            x = self.conv2(x, alphas, betas)
        else:
            x = self.conv2(x)
        x = self.conv3(x)
        if self.start_block:
            x = self.bn3(x)
        x = x + identity
        if self.end_block:
            x = self.bn3(x)
            x = self.act3(x)
        return x



def beta_softmax(betas, steps, scale=False):
    beta_list = []
    offset = 0
    for i in range(steps):
        beta = F.softmax(betas[offset:(offset + i + steps)], dim=0)
        if scale:
            beta = beta * len(beta)
        beta_list.append(beta)
        offset += i + steps
    betas = torch.cat(beta_list, dim=0)
    return betas


class Network(nn.Module):

    def __init__(self, depth, base_width=26, splits=4, num_classes=10, stages=(64, 64, 128, 256)):
        super().__init__()
        self.stages = stages
        self.splits = splits
        block = Bottleneck
        layers = [(depth - 2) // 9] * 3

        self.stem = Conv2d(3, self.stages[0], kernel_size=3, norm='def', act='def')
        self.in_channels = self.stages[0]

        self.layer1 = self._make_layer(
            block, self.stages[1], layers[0], stride=1,
            base_width=base_width, splits=splits)
        self.layer2 = self._make_layer(
            block, self.stages[2], layers[1], stride=2,
            base_width=base_width, splits=splits)
        self.layer3 = self._make_layer(
            block, self.stages[3], layers[2], stride=2,
            base_width=base_width, splits=splits)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.in_channels, num_classes)

        self._initialize_alphas()

    def _initialize_alphas(self):
        k = sum(4 + i for i in range(self.splits))
        num_ops = len(get_primitives())

        self.alphas = nn.Parameter(1e-3 * torch.randn(k), requires_grad=True)
        self.betas = nn.Parameter(1e-3 * torch.randn(self.splits, num_ops), requires_grad=True)

    def arch_parameters(self):
        return [self.alphas, self.betas]

    def model_parameters(self):
        ids = set(id(p) for p in self.arch_parameters())
        for p in self.parameters():
            if id(p) not in ids:
                yield p

    def _make_layer(self, block, channels, blocks, stride, **kwargs):
        layers = [block(self.in_channels, channels, stride=stride, start_block=True,
                        **kwargs)]
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels, stride=1,
                                exclude_bn0=i == 1, end_block=i == blocks - 1,
                                **kwargs))
        return nn.ModuleList(layers)

    def forward(self, x):
        alphas = beta_softmax(self.alphas, self.splits)
        betas = F.softmax(self.betas, dim=1)

        x = self.stem(x)

        for l in self.layer1:
            x = l(x, alphas, betas)
        for l in self.layer2:
            x = l(x, alphas, betas)
        for l in self.layer3:
            x = l(x, alphas, betas)

        x = self.avgpool(x)
        x = self.fc(x)
        return x


# class Network(nn.Module):
#
#     def __init__(self, C, layers, steps=4, multiplier=4, stem_multiplier=3, num_classes=10, cell_cls=Cell):
#         super().__init__()
#         self._C = C
#         self._num_classes = num_classes
#         self._steps = steps
#         self._multiplier = multiplier
#
#         C_curr = stem_multiplier * C
#         self.stem = nn.Sequential(
#             Conv2d(3, C_curr, kernel_size=3, bias=False),
#             Norm(C_curr, affine=True),
#         )
#
#         C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
#         self.cells = nn.ModuleList()
#         reduction_prev = False
#         for i in range(layers):
#             if i in [layers // 3, 2 * layers // 3]:
#                 C_curr *= 2
#                 reduction = True
#             else:
#                 reduction = False
#             cell = cell_cls(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
#             reduction_prev = reduction
#             self.cells.append(cell)
#             C_prev_prev, C_prev = C_prev, multiplier * C_curr
#
#         self.avg_pool = GlobalAvgPool()
#         self.classifier = nn.Linear(C_prev, num_classes)
#
#         self._initialize_alphas()
#
#     def _initialize_alphas(self):
#         k = sum(2 + i for i in range(self._steps))
#         num_ops = len(get_primitives())
#
#         self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops), requires_grad=True)
#         self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops), requires_grad=True)
#
#     def forward(self, x):
#         s0 = s1 = self.stem(x)
#         weights_reduce = F.softmax(self.alphas_reduce, dim=1)
#         weights_normal = F.softmax(self.alphas_normal, dim=1)
#         for cell in self.cells:
#             weights = weights_reduce if cell.reduction else weights_normal
#             s0, s1 = s1, cell(s0, s1, weights)
#         out = self.avg_pool(s1)
#         logits = self.classifier(out)
#         return logits
#
#     def arch_parameters(self):
#         return [self.alphas_normal, self.alphas_reduce]
#
#     def model_parameters(self):
#         ids = set(id(p) for p in self.arch_parameters())
#         for p in self.parameters():
#             if id(p) not in ids:
#                 yield p
#
#     def genotype(self):
#
#         gene_normal = parse_weights(F.softmax(self.alphas_normal.detach().cpu(), dim=1).numpy(), self._steps)
#         gene_reduce = parse_weights(F.softmax(self.alphas_reduce.detach().cpu(), dim=1).numpy(), self._steps)
#
#         concat = range(2 + self._steps - self._multiplier, self._steps + 2)
#         genotype = Genotype(
#             normal=gene_normal, normal_concat=concat,
#             reduce=gene_reduce, reduce_concat=concat
#         )
#         return genotype
#
#
# def parse_weights(weights, steps):
#     PRIMITIVES = get_primitives()
#
#     def get_op(w):
#         if 'none' in PRIMITIVES:
#             i = max([k for k in range(len(PRIMITIVES)) if k != PRIMITIVES.index('none')], key=lambda k: w[k])
#         else:
#             i = max(range(len(PRIMITIVES)), key=lambda k: w[k])
#         return w[i], PRIMITIVES[i]
#
#     gene = []
#     start = 0
#     for i in range(steps):
#         end = start + i + 2
#         W = weights[start:end]
#         edges = sorted(range(i + 2), key=lambda x: -get_op(W[x])[0])[:2]
#         for j in edges:
#             gene.append((get_op(W[j])[1], j))
#         start = end
#     return gene