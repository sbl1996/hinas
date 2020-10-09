from hinas.models.primitives import PRIMITIVES_darts
from hinas.train.darts.callbacks import PrintGenotype, TrainArchSchedule
from hinas.models.darts.search.pc_darts import Network

seed = 42
# train_batch_size = search_batch_size = val_batch_size = 256
train_batch_size = search_batch_size = val_batch_size = 2

primitives = PRIMITIVES_darts
# network_fn = lambda: Network(16, 8, k=4)
network_fn = lambda: Network(4, 5, k=2)

# epochs = 50
epochs = 10
arch_lr = 6e-4
model_lr = 0.1
grad_clip_norm = 5.0

work_dir = 'models/PC-DARTS'

val_freq = 5
callbacks = [
    PrintGenotype(from_epoch=5),
    TrainArchSchedule(after_epochs=5)
]