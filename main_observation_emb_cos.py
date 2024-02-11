import torch
import os 
import pdb
import sys
import numpy as np 
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForImageClassification





def cosine_metric(target_h_emb, eps=1e-8):
    # target_h_emb (B, Tokens, dims * heads)
    # normalize
    target_h_emb = target_h_emb.transpose(0,1)

    a_n = target_h_emb.norm(dim=1).unsqueeze(1)
    a_norm = target_h_emb / torch.max(a_n, eps * torch.ones_like(a_n))

    # patch-wise absolute value of cosine similarity
    sim_matrix = torch.einsum('bc,cd->bd', a_norm, a_norm.transpose(0,1))
    loss_cos = sim_matrix.abs().mean()

    return loss_cos

# for model_name in ['70m', '160m', '410m', '1b', '1.4b', '2.8b']:
#     for step in [0, 1, 2, 4, 8, 16, 32, 64, 128, 512, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 110000, 120000, 130000, 140000]:
#         model_checkpoint = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-{}'.format(model_name), revision='step{}'.format(step))
#         len_layers = len(model_checkpoint.gpt_neox.layers)
#         results = {}
#         for i in range(len_layers):
#             print(step, 'layer-{}'.format(i))
#             orthogonal = cosine_metric(model_checkpoint.gpt_neox.layers[i].mlp.dense_4h_to_h.weight)
#             results[i] = orthogonal
#         torch.save(results, 'pythia-{}-{}-rank.pt'.format(model_name, step))


# model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
# len_layers = len(model.model.layers)
# results = {}
# for i in range(len_layers):
#     print('layer-{}'.format(i))
#     orthogonal = cosine_metric(model.model.layers[i].mlp.down_proj.weight)
#     results[i] = orthogonal
# torch.save(results, 'llama-2-cos-weight.pt')


# model = torch.load('bert_seed0_step_2000000.pt')
# len_layers = 12
# results = {}
# for i in range(len_layers):
#     print('layer-{}'.format(i))
#     orthogonal = cosine_metric(model['bert.encoder.layer.{}.output.dense.weight'.format(i)])
#     results[i] = orthogonal
# torch.save(results, 'bert-base-cos-weight.pt')


# model = AutoModelForImageClassification.from_pretrained('google/vit-huge-patch14-224-in21k')

# len_layers = len(model.vit.encoder.layer)
# results = {}
# for i in range(len_layers):
#     print('layer-{}'.format(i))
#     orthogonal = cosine_metric(model.vit.encoder.layer[i].output.dense.weight)
#     results[i] = orthogonal
# torch.save(results, 'vit-huge-cos-weight.pt')


# model = AutoModelForCausalLM.from_pretrained('facebook/opt-6.7b')
# len_layers = len(model.model.decoder.layers)
# results = {}
# for i in range(len_layers):
#     print('layer-{}'.format(i))
#     orthogonal = cosine_metric(model.model.decoder.layers[i].fc2.weight)
#     results[i] = orthogonal
# torch.save(results, 'opt-6.7b-cos-weight.pt')