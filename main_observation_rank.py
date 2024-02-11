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


# for size in ['125m', '1.3b', '2.7b', '6.7b']:
#     model_checkpoint = AutoModelForCausalLM.from_pretrained('facebook/opt-{}'.format(size))
#     len_layers = len(model_checkpoint.model.decoder.layers)
#     stable_rank_results = {}
#     for i in range(len_layers):
#         print(size, 'layer-{}'.format(i))
#         rank_num = compute_weight_stable_rank(model_checkpoint.model.decoder.layers[i].fc1.weight)
#         stable_rank_results['layer-{}-fc1'.format(i)] = rank_num
#         rank_num = compute_weight_stable_rank(model_checkpoint.model.decoder.layers[i].fc2.weight)
#         stable_rank_results['layer-{}-fc2'.format(i)] = rank_num
#     torch.save(stable_rank_results, 'opt-{}-rank.pt'.format(size))


# for step in [10000, 20000, 30000, 40000, 50000, 150000, 200000, 250000]:
#     model_checkpoint = torch.load('hf_2.7b-{}/pytorch_model.bin'.format(step))
#     len_layers = 32
#     stable_rank_results = {}
#     for i in range(len_layers):
#         print(step, 'layer-{}'.format(i))
#         rank_num = compute_weight_stable_rank(model_checkpoint['decoder.layers.{}.fc1.weight'.format(i)].float())
#         stable_rank_results['layer-{}-fc1'.format(i)] = rank_num
#         rank_num = compute_weight_stable_rank(model_checkpoint['decoder.layers.{}.fc2.weight'.format(i)].float())
#         stable_rank_results['layer-{}-fc2'.format(i)] = rank_num
#     torch.save(stable_rank_results, 'opt-{}-rank.pt'.format(step))


# for step in [10000, 20000, 30000, 40000, 50000, 150000, 200000, 250000]:
#     model_checkpoint = torch.load('hf_2.7b-{}/pytorch_model.bin'.format(step))
#     len_layers = 32
#     stable_rank_results = {}
#     for i in range(len_layers):
#         print(step, 'layer-{}'.format(i))
#         rank_num = compute_weight_stable_rank(model_checkpoint['decoder.layers.{}.fc1.weight'.format(i)].float())
#         stable_rank_results['layer-{}-fc1'.format(i)] = rank_num
#         rank_num = compute_weight_stable_rank(model_checkpoint['decoder.layers.{}.fc2.weight'.format(i)].float())
#         stable_rank_results['layer-{}-fc2'.format(i)] = rank_num
#     torch.save(stable_rank_results, 'opt-{}-rank.pt'.format(step))


step_list = [0, 20000, 40000, 60000, 80000,
            100000, 120000, 140000, 160000, 180000, 200000,
            300000, 400000, 500000, 600000, 700000, 800000, 900000,
            1000000, 1100000, 1200000, 1300000, 1400000, 1500000,
            1600000, 1700000, 1800000, 1900000, 2000000]

for step in step_list:
    model_checkpoint = torch.load('bert_seed0_step_{}.pt'.format(step))
    len_layers = 12
    stable_rank_results = {}
    for i in range(len_layers):
        print(step, 'layer-{}'.format(i))
        rank_num = compute_weight_stable_rank(model_checkpoint['bert.encoder.layer.{}.intermediate.dense.weight'.format(i)].float())
        stable_rank_results['layer-{}-fc1'.format(i)] = rank_num
        rank_num = compute_weight_stable_rank(model_checkpoint['bert.encoder.layer.{}.output.dense.weight'.format(i)].float())
        stable_rank_results['layer-{}-fc2'.format(i)] = rank_num
    torch.save(stable_rank_results, 'bert-{}-rank.pt'.format(step))








