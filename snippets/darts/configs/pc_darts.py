from hinas.models.primitives import PRIMITIVES_darts
from hinas.train.darts.callbacks import PrintGenotype, TrainArch
from hinas.models.darts.search.pc_darts import Network

seed = 42
train_batch_size = search_batch_size = val_batch_size = 256

primitives = PRIMITIVES_darts
network_fn = lambda: Network(16, 8, k=4)

epochs = 50
arch_lr = 6e-4
model_lr = 0.1
grad_clip_norm = 5.0

work_dir = 'models/PC-DARTS'

val_freq = 5
callbacks = [
    PrintGenotype(from_epoch=15),
    TrainArch(after_epochs=15)
]