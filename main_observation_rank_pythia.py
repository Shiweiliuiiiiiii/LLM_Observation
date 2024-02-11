import torch
import os 
import pdb
import sys
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
from transformers import AutoModelForCausalLM


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

# for step in [0, 1, 2, 4, 8, 16, 32, 64, 128, 512, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 110000, 120000, 130000, 140000]:
#     model_checkpoint = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-70m', revision='step{}'.format(step))
#     len_layers = len(model_checkpoint.gpt_neox.layers)
#     stable_rank_results = {}
#     for i in range(len_layers):
#         print(step, 'layer-{}'.format(i))
#         rank_num = compute_weight_stable_rank(model_checkpoint.gpt_neox.layers[i].mlp.dense_4h_to_h.weight)
#         stable_rank_results['layer-{}-fc2'.format(i)] = rank_num
#         rank_num = compute_weight_stable_rank(model_checkpoint.gpt_neox.layers[i].mlp.dense_h_to_4h.weight)
#         stable_rank_results['layer-{}-fc1'.format(i)] = rank_num
#     torch.save(stable_rank_results, 'pythia-70m-{}-rank.pt'.format(step))

# for step in [0, 1, 2, 4, 8, 16, 32, 64, 128, 512, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 110000, 120000, 130000, 140000]:
#     model_checkpoint = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-1.4b', revision='step{}'.format(step), cache_dir=sys.argv[1])
#     len_layers = len(model_checkpoint.gpt_neox.layers)
#     stable_rank_results = {}
#     for i in range(len_layers):
#         print(step, 'layer-{}'.format(i))
#         rank_num = compute_weight_stable_rank(model_checkpoint.gpt_neox.layers[i].mlp.dense_4h_to_h.weight)
#         stable_rank_results['layer-{}-fc2'.format(i)] = rank_num
#         rank_num = compute_weight_stable_rank(model_checkpoint.gpt_neox.layers[i].mlp.dense_h_to_4h.weight)
#         stable_rank_results['layer-{}-fc1'.format(i)] = rank_num
#     torch.save(stable_rank_results, 'pythia-1.4b-{}-rank.pt'.format(step))

# for step in [0, 1, 2, 4, 8, 16, 32, 64, 128, 512, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 110000, 120000, 130000, 140000]:
#     model_checkpoint = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-12b', revision='step{}'.format(step), cache_dir=sys.argv[1])
#     len_layers = len(model_checkpoint.gpt_neox.layers)
#     stable_rank_results = {}
#     for i in range(len_layers):
#         print(step, 'layer-{}'.format(i))
#         rank_num = compute_weight_stable_rank(model_checkpoint.gpt_neox.layers[i].mlp.dense_4h_to_h.weight)
#         stable_rank_results['layer-{}-fc2'.format(i)] = rank_num
#         rank_num = compute_weight_stable_rank(model_checkpoint.gpt_neox.layers[i].mlp.dense_h_to_4h.weight)
#         stable_rank_results['layer-{}-fc1'.format(i)] = rank_num
#     torch.save(stable_rank_results, 'pythia-12b-{}-rank.pt'.format(step))

for step in [0, 1, 2, 4, 8, 16, 32, 64, 128, 512, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 110000, 120000, 130000, 140000]:
    model_checkpoint = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-6.9b', revision='step{}'.format(step), cache_dir=sys.argv[1])
    len_layers = len(model_checkpoint.gpt_neox.layers)
    stable_rank_results = {}
    for i in range(len_layers):
        print(step, 'layer-{}'.format(i))
        rank_num = compute_weight_stable_rank(model_checkpoint.gpt_neox.layers[i].mlp.dense_4h_to_h.weight)
        stable_rank_results['layer-{}-fc2'.format(i)] = rank_num
        rank_num = compute_weight_stable_rank(model_checkpoint.gpt_neox.layers[i].mlp.dense_h_to_4h.weight)
        stable_rank_results['layer-{}-fc1'.format(i)] = rank_num
    torch.save(stable_rank_results, 'pythia-6.9b-{}-rank.pt'.format(step))



