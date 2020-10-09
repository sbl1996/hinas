from hinas.models.primitives import PRIMITIVES_darts
from hinas.models.darts.search.darts import Network
from hinas.train.darts.callbacks import PrintGenotype

seed = 42
train_batch_size = search_batch_size = val_batch_size = 64

primitives = PRIMITIVES_darts
network_fn = lambda: Network(16, 8)

epochs = 50
arch_lr = 3e-4
model_lr = 0.025
model_wd = 3e-4 * 81
grad_clip_norm = 5.0

work_dir = 'models/RDARTS'

val_freq = 5
callbacks = [PrintGenotype(from_epoch=1)]