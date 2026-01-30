from .configuration_delta_net_2 import DeltaNet2Config
from .modeling_delta_net_2 import DeltaNet2ForCausalLM, DeltaNet2Model
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

__all__ = ['DeltaNet2Config', 'DeltaNet2ForCausalLM', 'DeltaNet2Model']

AutoConfig.register('delta_net_2', DeltaNet2Config)
AutoModel.register(DeltaNet2Config, DeltaNet2Model)
AutoModelForCausalLM.register(DeltaNet2Config, DeltaNet2ForCausalLM)