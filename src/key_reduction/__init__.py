from .pruners.rrqr import prune_model_strong_rrqr
from .pruners.wanda import prune_model_wanda
from .pruners.grad import prune_model as prune_model_grad
from .pruners.pca import absorb_and_compress_layer as prune_model_pca
from .pruners.l1 import prune_model_l1
from .pruners.random import prune_model_random
