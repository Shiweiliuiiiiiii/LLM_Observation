import torch
import os 
import pdb
import numpy as np 


def compute_weight_stable_rank(weight: torch.Tensor) -> float:
    """
    Compute the stable rank of a weight matrix

    Parameters
    ----------
    weight: torch.Tensor
        The weight matrix

    Returns
    -------
    float
        The stable rank of the weight matrix

    Examples
    --------
    >>> import torch
    >>> compute_weight_stable_rank(torch.randn(100, 100)) > 20
    True
    """
    singular_values = torch.linalg.svdvals(weight)
    return torch.sum(singular_values ** 2).item() / (torch.max(singular_values).item() ** 2 + 1e-8)

# llama-2-7b
from transformers import AutoModelForCausalLM, AutoConfig


# model_checkpoint = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
# stable_rank_results = {}
# for i in range(32):
#     rank_num = compute_weight_stable_rank(model_checkpoint.model.layers[i].mlp.gate_proj.weight)
#     stable_rank_results['layer-{}-gate_proj'.format(i)] = rank_num
#     rank_num = compute_weight_stable_rank(model_checkpoint.model.layers[i].mlp.down_proj.weight)
#     stable_rank_results['layer-{}-down_proj'.format(i)] = rank_num
#     rank_num = compute_weight_stable_rank(model_checkpoint.model.layers[i].mlp.up_proj.weight)
#     stable_rank_results['layer-{}-up_proj'.format(i)] = rank_num
# torch.save(stable_rank_results, 'llama-2-7b-rank.pt')



# config_name = 'facebook/opt-2.7b'
# config = AutoConfig.from_pretrained(config_name)
# model = AutoModelForCausalLM.from_config(config)
# len_layers = len(model.model.decoder.layers)
# print(config_name)
# for idx in range(len_layers):
#     stable_rank = compute_weight_stable_rank(model.model.decoder.layers[idx].fc1.weight)
#     print('Rank = {}'.format(stable_rank))

# config_name = 'EleutherAI/pythia-70m'
# config = AutoConfig.from_pretrained(config_name)
# model = AutoModelForCausalLM.from_config(config)
# len_layers = len(model.gpt_neox.layers)
# print(config_name)
# for idx in range(len_layers):
#     stable_rank = compute_weight_stable_rank(model.gpt_neox.layers[idx].mlp.dense_h_to_4h.weight)
#     print('Rank = {}'.format(stable_rank))

# config_name = 'EleutherAI/pythia-1.4b'
# config = AutoConfig.from_pretrained(config_name)
# model = AutoModelForCausalLM.from_config(config)
# len_layers = len(model.gpt_neox.layers)
# print(config_name)
# for idx in range(len_layers):
#     stable_rank = compute_weight_stable_rank(model.gpt_neox.layers[idx].mlp.dense_h_to_4h.weight)
#     print('Rank = {}'.format(stable_rank))

# config_name = 'EleutherAI/pythia-12b'
# config = AutoConfig.from_pretrained(config_name)
# model = AutoModelForCausalLM.from_config(config)
# len_layers = len(model.gpt_neox.layers)
# print(config_name)
# for idx in range(len_layers):
#     stable_rank = compute_weight_stable_rank(model.gpt_neox.layers[idx].mlp.dense_h_to_4h.weight)
#     print('Rank = {}'.format(stable_rank))


config_name = 'EleutherAI/pythia-6.9b'
config = AutoConfig.from_pretrained(config_name)
model = AutoModelForCausalLM.from_config(config)
len_layers = len(model.gpt_neox.layers)
print(config_name)
for idx in range(len_layers):
    stable_rank = compute_weight_stable_rank(model.gpt_neox.layers[idx].mlp.dense_h_to_4h.weight)
    print('Rank = {}'.format(stable_rank))



