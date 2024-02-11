import torch 
import numpy as np 
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt 

# layer_list = [0, 1, 2, 3]
# layer_list = [13,14,15,16]
# layer_list = [28,29,30,31]
# all_layer_results = []
# for layer in layer_list:
#     results = []
#     for iteration in [1,2,3,4,5,15,20,25]:
#         data = torch.load('opt-2.7b-{}w-wiki103-attn_entropy.pt'.format(iteration), map_location='cpu')[layer]
#         results.append(np.mean(data[layer]))
#     all_layer_results.append(results)

# iterations = [10000, 20000, 30000, 40000, 50000, 150000, 200000, 250000]
# ppl = [43.07, 36.69, 34.58, 33.13, 32.13, 28.49, 27.86, 26.43]


# plt.plot(iterations, ppl, linestyle='dashdot', color='black', label='PPL')
# plt.ylabel('PPL')
# plt.legend(loc='lower left')
# plt.twinx()
# for i in range(len(layer_list)):
#     plt.plot(iterations, all_layer_results[i], label='Layer-{}'.format(layer_list[i]+1))
# plt.ylabel('Entropy')
# plt.xlabel('Iterations')
# plt.title('OPT-2.7B Late Layers')
# plt.legend()
# plt.show()





# #####################################
# # show rank across different layers #
# #####################################

# # data = torch.load('llama-2-7b-rank.pt')
# # print('Layer    Gate    Down    Up    Average')
# # for i in range(32):
# #     print('{} \t {:.0f} \t {:.0f} \t {:.0f} \t {:.0f}'.format(i+1, data['layer-{}-gate_proj'.format(i)], data['layer-{}-down_proj'.format(i)], data['layer-{}-up_proj'.format(i)], (data['layer-{}-gate_proj'.format(i)]+data['layer-{}-down_proj'.format(i)]+data['layer-{}-up_proj'.format(i)])/3))



# # data = torch.load('opt-125m-rank.pt')
# # print('Layer    fc1    fc2   Average')
# # for i in range(12):
# #     print('{} \t {:.0f} \t {:.0f} \t {:.0f}'.format(i+1, data['layer-{}-fc1'.format(i)], data['layer-{}-fc2'.format(i)], (data['layer-{}-fc1'.format(i)]+data['layer-{}-fc2'.format(i)])/2))


# # data = torch.load('opt-1.3b-rank.pt')
# # print('Layer    fc1    fc2   Average')
# # for i in range(24):
# #     print('{} \t {:.0f} \t {:.0f} \t {:.0f}'.format(i+1, data['layer-{}-fc1'.format(i)], data['layer-{}-fc2'.format(i)], (data['layer-{}-fc1'.format(i)]+data['layer-{}-fc2'.format(i)])/2))



# #####################################
# ##### Attention Entropy in OPT ######
# #####################################
# plt.figure(figsize=(20,3.5))

# plt.subplot(1,4,3)
# nlayer = 32
# all_layer_results = []
# layer_list = [i for i in range(nlayer)]
# for layer in layer_list:
#     results = []
#     for iteration in [0,1,2,3,4,5,15,20,25]:
#         if iteration == 0:
#             data = torch.load('opt-2.7b-wiki103-random-attn_entropy.pt', map_location='cpu')[layer]
#         else:
#             data = torch.load('opt-2.7b-{}w-wiki103-attn_entropy.pt'.format(iteration), map_location='cpu')[layer]
#         results.append(np.mean(data))
#     all_layer_results.append(results)

# iterations = [0, 10, 20, 30, 40, 50, 150, 200, 250]
# ppl = [43.07, 36.69, 34.58, 33.13, 32.13, 28.49, 27.86, 26.43]

# cmaps = plt.get_cmap('rainbow', nlayer)
# for i in range(len(layer_list)):
#     plt.plot(iterations, all_layer_results[i], label='layer{}'.format(layer_list[i]), color=cmaps(i/nlayer))
# plt.ylabel("Attention Entropy")
# plt.xlabel("Minibatch (k)")
# plt.title('OPT-2.7B, Layers: 32, val_loss: 3.274')
# norm_iters = plt.Normalize(0, nlayer - 1)
# sm = cm.ScalarMappable(cmap=cmaps, norm=norm_iters)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=plt.gca())
# cbar.set_label('', rotation=270, labelpad=15)
# # plt.legend()
# # plt.show()



# ####################################
# #### Attention Entropy in Pythia ###
# ####################################

# plt.subplot(1,4,1)
# nlayer = 6

# all_layer_results = []
# layer_list = [i for i in range(nlayer)]
# for layer in layer_list:
#     results = []
#     for iteration in [0,1,2,4,8,16,32,64,128,512,1000,2000,5000,10000,20000,50000,100000,110000,120000,130000,140000]:
#         data = torch.load('pythia_70m_wiki103_attn_entropy_step{}-attn_entropy.pt'.format(iteration), map_location='cpu')[layer]
#         results.append(np.mean(data))
#     all_layer_results.append(results)

# iterations = [0,1,2,4,8,16,32,64,128,512,1000,2000,5000,10000,20000,50000,100000,110000,120000,130000,140000]
# k_iterations = [item/1000 for item in iterations]

# cmaps = plt.get_cmap('rainbow', nlayer)
# for i in range(len(layer_list)):
#     plt.plot(k_iterations, all_layer_results[i], label='layer{}'.format(layer_list[i]), color=cmaps(i/nlayer))
# plt.ylabel("Attention Entropy")
# plt.xlabel("Minibatch (k)")
# plt.title('Pythia-70M, Layers: 6, val_loss: 4.021')
# norm_iters = plt.Normalize(0, nlayer - 1)
# sm = cm.ScalarMappable(cmap=cmaps, norm=norm_iters)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=plt.gca())
# cbar.set_label('', rotation=270, labelpad=15)
# # plt.legend()
# # plt.show()



# ##############################################
# ###  MLP rank in OPT ###
# ##############################################

# random_rank = np.loadtxt('opt_random_rank.txt')

# plt.subplot(1,4,4)
# nlayer = 32
# all_layer_results_rank = []
# layer_list = [i for i in range(nlayer)]
# for layer in layer_list:
#     results_rank = []
#     for iteration in [0,1,2,3,4,5,15,20,25]:
#         if iteration == 0:
#             results_rank.append(random_rank[layer]/2560)
#         else:
#             data_rank = torch.load('opt-{}0000-rank.pt'.format(iteration), map_location='cpu')['layer-{}-fc1'.format(layer)]
#             results_rank.append(data_rank / 2560)
#     all_layer_results_rank.append(results_rank)

# iterations = [0, 10, 20, 30, 40, 50, 150, 200, 250]

# cmaps = plt.get_cmap('rainbow', nlayer)
# for i in range(len(layer_list)):
#     plt.plot(iterations, all_layer_results_rank[i], linestyle='dashdot', label='layer{}'.format(layer_list[i]), color=cmaps(i/nlayer))
# plt.ylabel("# Stable Rank")
# plt.xlabel("Minibatch (k)")
# plt.title('OPT-2.7B, Layers: 32, val_loss: 3.274')
# norm_iters = plt.Normalize(0, nlayer - 1)
# sm = cm.ScalarMappable(cmap=cmaps, norm=norm_iters)
# sm.set_array([])
# plt.ylim(0.005, 0.04)
# cbar = plt.colorbar(sm, ax=plt.gca())
# cbar.set_label('', rotation=270, labelpad=15)
# # plt.legend()
# # plt.show()



# #############################################
# ##  MLP rank in Pythia ###
# #############################################

# plt.subplot(1,4,2)
# nlayer = 6
# all_layer_results = []
# all_layer_results_rank = []
# layer_list = [i for i in range(nlayer)]
# for layer in layer_list:
#     results = []
#     results_rank = []
#     for iteration in [0,1,2,4,8,16,32,64,128,512,1000,2000,5000,10000,20000,50000,100000,110000,120000,130000,140000]:
#         data = torch.load('pythia_70m_wiki103_attn_entropy_step{}-attn_entropy.pt'.format(iteration), map_location='cpu')[layer]
#         results.append(np.mean(data[layer]))

#         data_rank = torch.load('pythia-70m-{}-rank.pt'.format(iteration), map_location='cpu')['layer-{}-fc1'.format(layer)]
#         results_rank.append(data_rank / 512)
#     all_layer_results.append(results)
#     all_layer_results_rank.append(results_rank)

# iterations = [0,1,2,4,8,16,32,64,128,512,1000,2000,5000,10000,20000,50000,100000,110000,120000,130000,140000]
# k_iterations = [item/1000 for item in iterations]

# cmaps = plt.get_cmap('rainbow', nlayer)
# for i in range(len(layer_list)):
#     plt.plot(k_iterations, all_layer_results_rank[i], linestyle='dashdot', label='layer{}'.format(layer_list[i]), color=cmaps(i/nlayer))
# plt.ylabel("# Stable Rank")
# plt.xlabel("Minibatch (k)")
# plt.title('Pythia-70M, Layers: 6, val_loss: 4.021')
# norm_iters = plt.Normalize(0, nlayer - 1)
# sm = cm.ScalarMappable(cmap=cmaps, norm=norm_iters)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=plt.gca())
# cbar.set_label('', rotation=270, labelpad=15)
# plt.ylim(0,0.15)
# # plt.legend()
# # plt.show()
# plt.savefig('opt_pythia_results.pdf', bbox_inches='tight')
# plt.close()



















# ####################################
# #### Attention Entropy in Pythia-1.4b ###
# ####################################
# plt.figure(figsize=(20,3.5))
# plt.subplot(1,4,1)
# nlayer = 24

# all_layer_results = []
# layer_list = [i for i in range(nlayer)]
# for layer in layer_list:
#     results = []
#     for iteration in [0,1,2,4,8,16,32,64,128,512,1000,2000,5000,10000,20000,50000,100000,110000,120000,130000,140000]:
#         data = torch.load('pythia_1.4b_wiki103_attn_entropy_step{}-attn_entropy.pt'.format(iteration), map_location='cpu')[layer]
#         results.append(np.mean(data))
#     all_layer_results.append(results)

# iterations = [0,1,2,4,8,16,32,64,128,512,1000,2000,5000,10000,20000,50000,100000,110000,120000,130000,140000]
# k_iterations = [item/1000 for item in iterations]

# cmaps = plt.get_cmap('rainbow', nlayer)
# for i in range(len(layer_list)):
#     plt.plot(k_iterations, all_layer_results[i], label='layer{}'.format(layer_list[i]), color=cmaps(i/nlayer))
# plt.ylabel("Attention Entropy")
# plt.xlabel("Minibatch (k)")
# plt.title('Pythia-1.4B, Layers: 24, val_loss: 2.625')
# norm_iters = plt.Normalize(0, nlayer - 1)
# sm = cm.ScalarMappable(cmap=cmaps, norm=norm_iters)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=plt.gca())
# cbar.set_label('', rotation=270, labelpad=15)
# # plt.legend()
# # plt.show()



# ####################################
# #### Attention Entropy in Pythia 6.9B ###
# ####################################

# plt.subplot(1,4,3)
# nlayer = 32

# all_layer_results = []
# layer_list = [i for i in range(nlayer)]
# for layer in layer_list:
#     results = []
#     for iteration in [0,1,2,4,8,16,32,64,128,512,1000,2000,5000,10000,20000,50000,100000,110000,120000,130000,140000]:
#         data = torch.load('pythia_6.9b_wiki103_attn_entropy_step{}-attn_entropy.pt'.format(iteration), map_location='cpu')[layer]
#         results.append(np.mean(data))
#     all_layer_results.append(results)

# iterations = [0,1,2,4,8,16,32,64,128,512,1000,2000,5000,10000,20000,50000,100000,110000,120000,130000,140000]
# k_iterations = [item/1000 for item in iterations]

# cmaps = plt.get_cmap('rainbow', nlayer)
# for i in range(len(layer_list)):
#     plt.plot(k_iterations, all_layer_results[i], label='layer{}'.format(layer_list[i]), color=cmaps(i/nlayer))
# plt.ylabel("Attention Entropy")
# plt.xlabel("Minibatch (k)")
# plt.title('Pythia-6.9B, Layers: 32, val_loss: 2.358')
# norm_iters = plt.Normalize(0, nlayer - 1)
# sm = cm.ScalarMappable(cmap=cmaps, norm=norm_iters)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=plt.gca())
# cbar.set_label('', rotation=270, labelpad=15)
# # plt.legend()
# # plt.show()



# ##############################################
# ###  MLP rank in Pythia-1.4b ###
# ##############################################


# plt.subplot(1,4,2)
# nlayer = 24
# all_layer_results_rank = []
# layer_list = [i for i in range(nlayer)]
# for layer in layer_list:
#     results_rank = []
#     for iteration in [0,1,2,4,8,16,32,64,128,512,1000,2000,5000,10000,20000,50000,100000,110000,120000,130000,140000]:
#         data_rank = torch.load('pythia-1.4b-{}-rank.pt'.format(iteration), map_location='cpu')['layer-{}-fc1'.format(layer)]
#         results_rank.append(data_rank / 2048)
#     all_layer_results_rank.append(results_rank)

# iterations = [0,1,2,4,8,16,32,64,128,512,1000,2000,5000,10000,20000,50000,100000,110000,120000,130000,140000]
# k_iterations = [item/1000 for item in iterations]

# cmaps = plt.get_cmap('rainbow', nlayer)
# for i in range(len(layer_list)):
#     plt.plot(k_iterations, all_layer_results_rank[i], linestyle='dashdot', label='layer{}'.format(layer_list[i]), color=cmaps(i/nlayer))
# plt.ylabel("# Stable Rank")
# plt.xlabel("Minibatch (k)")
# plt.title('Pythia-1.4B, Layers: 24, val_loss: 2.625')

# plt.ylim(0, 0.06)
# norm_iters = plt.Normalize(0, nlayer - 1)
# sm = cm.ScalarMappable(cmap=cmaps, norm=norm_iters)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=plt.gca())
# cbar.set_label('', rotation=270, labelpad=15)


# #############################################
# ##  MLP rank in Pythia-6.9b ###
# #############################################
# plt.subplot(1,4,4)
# nlayer = 32
# all_layer_results_rank = []
# layer_list = [i for i in range(nlayer)]
# for layer in layer_list:
#     results_rank = []
#     for iteration in [0,1,2,4,8,16,32,64,128,512,1000,2000,5000,10000,20000,50000,100000,110000,120000,130000,140000]:
#         data_rank = torch.load('pythia-6.9b-{}-rank.pt'.format(iteration), map_location='cpu')['layer-{}-fc1'.format(layer)]
#         results_rank.append(data_rank / 4096)
#     all_layer_results_rank.append(results_rank)

# iterations = [0,1,2,4,8,16,32,64,128,512,1000,2000,5000,10000,20000,50000,100000,110000,120000,130000,140000]
# k_iterations = [item/1000 for item in iterations]

# cmaps = plt.get_cmap('rainbow', nlayer)
# for i in range(len(layer_list)):
#     plt.plot(k_iterations, all_layer_results_rank[i], linestyle='dashdot', label='layer{}'.format(layer_list[i]), color=cmaps(i/nlayer))
# plt.ylabel("# Stable Rank")
# plt.xlabel("Minibatch (k)")
# plt.title('Pythia-6.9B, Layers: 32, val_loss: 2.358')
# norm_iters = plt.Normalize(0, nlayer - 1)
# sm = cm.ScalarMappable(cmap=cmaps, norm=norm_iters)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=plt.gca())
# cbar.set_label('', rotation=270, labelpad=15)
# plt.ylim(0.009, 0.035)
# # plt.legend()
# # plt.show()
# plt.savefig('opt_pythia_results_2.pdf', bbox_inches='tight')
# plt.close()






# #####################################
# ##### Attention Entropy in BERT ######
# #####################################
# plt.figure(figsize=(5,3.5))

# # plt.subplot(1,2,1)
# nlayer = 12
# all_layer_results = []
# layer_list = [i for i in range(nlayer)]
# for layer in layer_list:
#     results = []
#     for iteration in [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000,
#                     180000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000,
#                     1000000, 1100000, 1200000, 1300000, 1400000, 1500000,
#                     1600000, 1700000, 1800000, 1900000, 2000000]:
#         data = torch.load('bert_base_wiki103_step_{}-attn_entropy.pt'.format(iteration), map_location='cpu')[layer]
#         results.append(np.mean(data))
#     all_layer_results.append(results)

# iterations = [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000,
#                     180000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000,
#                     1000000, 1100000, 1200000, 1300000, 1400000, 1500000,
#                     1600000, 1700000, 1800000, 1900000, 2000000]
# iterations = [item/1000 for item in iterations]

# cmaps = plt.get_cmap('rainbow', nlayer)
# for i in range(len(layer_list)):
#     plt.plot(iterations, all_layer_results[i], label='layer{}'.format(layer_list[i]), color=cmaps(i/nlayer))
# plt.ylabel("Attention Entropy")
# plt.xlabel("Minibatch (k)")
# plt.title('BERT, Layers: 12, val_loss: 1.517')
# norm_iters = plt.Normalize(0, nlayer - 1)
# sm = cm.ScalarMappable(cmap=cmaps, norm=norm_iters)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=plt.gca())
# cbar.set_label('', rotation=270, labelpad=15)
# plt.savefig('bert_attn.svg', bbox_inches='tight')
# plt.close()
# # plt.legend()
# plt.show()
#####################################
########### Rank in BERT ############
#####################################
# plt.figure(figsize=(5,3.5))
# nlayer = 12
# all_layer_results_rank = []
# layer_list = [i for i in range(nlayer)]
# for layer in layer_list:
#     results_rank = []
#     for iteration in [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000,
#                     180000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000,
#                     1000000, 1100000, 1200000, 1300000, 1400000, 1500000,
#                     1600000, 1700000, 1800000, 1900000, 2000000]:
#         data_rank = torch.load('bert-{}-rank.pt'.format(iteration), map_location='cpu')['layer-{}-fc1'.format(layer)]
#         results_rank.append(data_rank / 768)
#     all_layer_results_rank.append(results_rank)

# iterations = [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000,
#                     180000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000,
#                     1000000, 1100000, 1200000, 1300000, 1400000, 1500000,
#                     1600000, 1700000, 1800000, 1900000, 2000000]
# iterations = [item/1000 for item in iterations]

# cmaps = plt.get_cmap('rainbow', nlayer)
# for i in range(len(layer_list)):
#     plt.plot(iterations, all_layer_results_rank[i], linestyle='dashdot', label='layer{}'.format(layer_list[i]), color=cmaps(i/nlayer))
# plt.ylabel("# Stable Rank")
# plt.xlabel("Minibatch (k)")
# plt.title('BERT, Layers: 12, val_loss: 1.517')

# plt.ylim(0, 0.1)
# norm_iters = plt.Normalize(0, nlayer - 1)
# sm = cm.ScalarMappable(cmap=cmaps, norm=norm_iters)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=plt.gca())
# cbar.set_label('', rotation=270, labelpad=15)
# # plt.legend()
# plt.savefig('bert_rank.svg', bbox_inches='tight')
# plt.close()



##################################
## BERT EMB ORTH in BERT ######
# ####################################
# plt.figure(figsize=(5,3.5))
# # plt.subplot(1,4,1)
# nlayer = 12
# all_layer_results = []
# layer_list = [i for i in range(nlayer)]
# for layer in layer_list:
#     results = []
#     for iteration in [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000,
#                     180000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000,
#                     1000000, 1100000, 1200000, 1300000, 1400000, 1500000,
#                     1600000, 1700000, 1800000, 1900000, 2000000]:
#         data = torch.load('bert_base_wiki103_step_{}-emb_before-abs.pt'.format(iteration), map_location='cpu')[layer]
#         results.append(np.mean(data))
#     all_layer_results.append(results)

# iterations = [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000,
#                     180000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000,
#                     1000000, 1100000, 1200000, 1300000, 1400000, 1500000,
#                     1600000, 1700000, 1800000, 1900000, 2000000]
# iterations = [item/1000 for item in iterations]

# cmaps = plt.get_cmap('rainbow', nlayer)
# for i in range(len(layer_list)):
#     plt.plot(iterations, all_layer_results[i], label='layer{}'.format(layer_list[i]), color=cmaps(i/nlayer))
#     plt.scatter(iterations[0], all_layer_results[i][0], marker='*', color=cmaps(i/nlayer), s=50)
# plt.ylabel("Cosine Similarity")
# plt.xlabel("Minibatch (k)")
# # plt.title('BERT, Layers: 12, val_loss: 1.517')
# norm_iters = plt.Normalize(0, nlayer - 1)
# sm = cm.ScalarMappable(cmap=cmaps, norm=norm_iters)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=plt.gca())
# cbar.set_label('', rotation=270, labelpad=15)
# plt.savefig('bert_cos_sup.svg', bbox_inches='tight')
# plt.close()
# # plt.legend()
# plt.show()

# plt.figure(figsize=(5,3.5))
# # plt.subplot(1,4,1)
# nlayer = 12
# all_layer_results = []
# layer_list = [i for i in range(nlayer)]
# for layer in layer_list:
#     results = []
#     for iteration in [0, 20000, 40000]:
#         data = torch.load('bert_base_wiki103_step_{}-emb_before-abs.pt'.format(iteration), map_location='cpu')[layer]
#         results.append(np.mean(data))
#     all_layer_results.append(results)

# iterations = [0, 20000, 40000]
# iterations = [item/1000 for item in iterations]

# cmaps = plt.get_cmap('rainbow', nlayer)
# for i in range(len(layer_list)):
#     plt.plot(iterations, all_layer_results[i], label='layer{}'.format(layer_list[i]), color=cmaps(i/nlayer))
#     plt.scatter(iterations[0], all_layer_results[i][0], marker='*', color=cmaps(i/nlayer), s=50)
# plt.ylabel("Cosine Similarity")
# plt.xlabel("Minibatch (k)")
# # plt.title('BERT, Layers: 12, val_loss: 1.517')
# norm_iters = plt.Normalize(0, nlayer - 1)
# sm = cm.ScalarMappable(cmap=cmaps, norm=norm_iters)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=plt.gca())
# cbar.set_label('', rotation=270, labelpad=15)
# plt.savefig('bert_cos_sup.svg', bbox_inches='tight')
# plt.close()
# # plt.legend()
# # plt.show()


# plt.figure(figsize=(20,3.5))
# nlayer = 6
# all_layer_results = []
# layer_list = [i for i in range(nlayer)]
# for layer in layer_list:
#     results = []
#     for iteration in [0,1,2,4,8,16,32,64,128,512,1000,2000,5000,10000,20000,50000,100000,110000,120000,130000,140000]:
#         data = torch.load('pythia_70m_wiki103_emb_step{}-emb_before-abs.pt'.format(iteration), map_location='cpu')[layer]
#         results.append(np.mean(data))
#     all_layer_results.append(results)

# iterations = [0,1,2,4,8,16,32,64,128,512,1000,2000,5000,10000,20000,50000,100000,110000,120000,130000,140000]
# k_iterations = [item/1000 for item in iterations]

# cmaps = plt.get_cmap('rainbow', nlayer)
# for i in range(len(layer_list)):
#     plt.plot(k_iterations, all_layer_results[i], label='layer{}'.format(layer_list[i]), color=cmaps(i/nlayer))
#     plt.scatter(k_iterations[0], all_layer_results[i][0], marker='*', color=cmaps(i/nlayer), s=50)
# plt.ylabel("Cosine Similarity")
# plt.xlabel("Minibatch (k)")
# plt.title('Pythia-70m, Layers: 6, val_loss: 4.021')
# norm_iters = plt.Normalize(0, nlayer - 1)
# sm = cm.ScalarMappable(cmap=cmaps, norm=norm_iters)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=plt.gca())
# cbar.set_label('', rotation=270, labelpad=15)
# plt.savefig('pythia_70m_cos.svg', bbox_inches='tight')
# plt.close()


# plt.legend()
# plt.show()
# plt.savefig('cos_results.pdf', bbox_inches='tight')
# plt.close()


# plt.figure(figsize=(20,3.5))
# nlayer = 24
# all_layer_results = []
# layer_list = [i for i in range(nlayer)]
# for layer in layer_list:
#     results = []
#     for iteration in [0,1,2,4,8,16,32,64,128,512,1000,2000,5000,10000,20000,50000,100000,110000,120000,130000,140000]:
#         data = torch.load('pythia_1.4b_wiki103_emb_step{}-emb_before-abs.pt'.format(iteration), map_location='cpu')[layer]
#         results.append(np.mean(data))
#     all_layer_results.append(results)

# iterations = [0,1,2,4,8,16,32,64,128,512,1000,2000,5000,10000,20000,50000,100000,110000,120000,130000,140000]
# k_iterations = [item/1000 for item in iterations]

# cmaps = plt.get_cmap('rainbow', nlayer)
# for i in range(len(layer_list)):
#     plt.plot(k_iterations, all_layer_results[i], label='layer{}'.format(layer_list[i]), color=cmaps(i/nlayer))
#     plt.scatter(k_iterations[0], all_layer_results[i][0], marker='*', color=cmaps(i/nlayer), s=50)
# plt.ylabel("Cosine Similarity")
# plt.xlabel("Minibatch (k)")
# plt.title('Pythia-1.4b, Layers: 24, val_loss: 2.625')
# norm_iters = plt.Normalize(0, nlayer - 1)
# sm = cm.ScalarMappable(cmap=cmaps, norm=norm_iters)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=plt.gca())
# cbar.set_label('', rotation=270, labelpad=15)
# plt.savefig('pythia_1.4b_cos.svg', bbox_inches='tight')
# plt.close()



# opt = torch.load('opt_6.7b_wiki103_emb-emb_before-abs.pt')
# llama = torch.load('llama_2_7b_wiki103_emb-emb_before-abs.pt')
# vit = torch.load('beans_obs_emb_vit_huge-emb_before-abs.pt')
# bert = torch.load('bert_base_wiki103_step_2000000-emb_before-abs.pt')

# def mean_results(data):
#     meandata = []
#     nlayer = len(data.keys())
#     for layer in range(nlayer):
#         meandata.append(np.mean(data[layer]))
#     return meandata

# plt.figure(figsize=(4,3.5))
# plt.grid(linestyle='dashed', zorder=0)
# opt = mean_results(opt)
# llama = mean_results(llama)
# vit = mean_results(vit)
# bert = mean_results(bert)

# plt.plot(bert, color='red', label='BERT-Base')
# plt.plot(llama, color='forestgreen', label='OPT-6.7B')
# plt.plot(vit, color='darkorange', label='LLAMA-2-7B')
# plt.plot(opt, color='royalblue', label='ViT-Huge')
# plt.xlabel('Layer Index')
# plt.ylabel('Cosine Similarity')
# plt.title('Final Checkpoint')
# plt.legend()
# plt.savefig('other_arch.svg', bbox_inches='tight')
# plt.close()








# layer_list = [6,6,12,12,24,24,16,16,24,24,32,32,32]

# for model_name, nlayer in zip(['pythia_70m', 'pythia_70m-deduped', 'pythia_160m', 'pythia_160m-deduped',
#                     'pythia_410m', 'pythia_410m-deduped', 'pythia_1b', 'pythia_1b-deduped',
#                     'pythia_1.4b', 'pythia_1.4b-deduped', 'pythia_2.8b', 'pythia_2.8b-deduped', 'pythia_6.9b'], layer_list):
#     plt.figure(figsize=(5,3.5))
#     all_layer_results = []
#     layer_list = [i for i in range(nlayer)]
#     for layer in layer_list:
#         results = []
#         for iteration in [0,1,2,4,8,16,32,64,128,512,1000,2000,5000,10000,20000,50000,100000,110000]:
#             data = torch.load('{}_wiki103_emb_step{}-emb_before-abs.pt'.format(model_name, iteration), map_location='cpu')[layer]
#             results.append(np.mean(data))
#         all_layer_results.append(results)

#     iterations = [0,1,2,4,8,16,32,64,128,512,1000,2000,5000,10000,20000,50000,100000,110000]
#     k_iterations = [item/1000 for item in iterations]

#     cmaps = plt.get_cmap('rainbow', nlayer)
#     for i in range(len(layer_list)):
#         plt.plot(k_iterations, all_layer_results[i], label='layer{}'.format(layer_list[i]), color=cmaps(i/nlayer))
#         plt.scatter(k_iterations[0], all_layer_results[i][0], marker='*', color=cmaps(i/nlayer), s=50)
#     plt.ylabel("Cosine Similarity")
#     plt.xlabel("Minibatch (k)")
#     # plt.title('Pythia-1.4b, Layers: 24, val_loss: 2.625')
#     norm_iters = plt.Normalize(0, nlayer - 1)
#     sm = cm.ScalarMappable(cmap=cmaps, norm=norm_iters)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=plt.gca())
#     cbar.set_label('', rotation=270, labelpad=15)
#     plt.savefig('{}_cos.svg'.format(model_name), bbox_inches='tight')
#     plt.close()



# # plt.show()



# layer_list = [6,6,12,12,24,24,16,16,24,24,32,32,32]

# for model_name, nlayer in zip(['pythia_70m', 'pythia_70m-deduped', 'pythia_160m', 'pythia_160m-deduped',
#                     'pythia_410m', 'pythia_410m-deduped', 'pythia_1b', 'pythia_1b-deduped',
#                     'pythia_1.4b', 'pythia_1.4b-deduped', 'pythia_2.8b', 'pythia_2.8b-deduped', 'pythia_6.9b'], layer_list):
#     plt.figure(figsize=(5,3.5))
#     all_layer_results = []
#     layer_list = [i for i in range(nlayer)]
#     for layer in layer_list:
#         results = []
#         for iteration in [0,1,2,4,8,16,32,64,128]:
#             data = torch.load('{}_wiki103_emb_step{}-emb_before-abs.pt'.format(model_name, iteration), map_location='cpu')[layer]
#             results.append(np.mean(data))
#         all_layer_results.append(results)

#     iterations = [0,1,2,4,8,16,32,64,128]

#     cmaps = plt.get_cmap('rainbow', nlayer)
#     for i in range(len(layer_list)):
#         plt.plot(iterations, all_layer_results[i], label='layer{}'.format(layer_list[i]), color=cmaps(i/nlayer))
#         plt.scatter(iterations[0], all_layer_results[i][0], marker='*', color=cmaps(i/nlayer), s=50)
#     plt.ylabel("Cosine Similarity")
#     plt.xlabel("Minibatch")

#     # plt.title('Pythia-1.4b, Layers: 24, val_loss: 2.625')
#     norm_iters = plt.Normalize(0, nlayer - 1)
#     sm = cm.ScalarMappable(cmap=cmaps, norm=norm_iters)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=plt.gca())
#     cbar.set_label('', rotation=270, labelpad=15)
#     plt.savefig('{}_cos_supres.svg'.format(model_name), bbox_inches='tight')
#     plt.close()







# layer_list = [6,12,24,16,24,32]

# for model_name, nlayer in zip(['70m', '160m', '410m', '1b', '1.4b', '2.8b'], layer_list):
#     plt.figure(figsize=(5,3.5))
#     all_layer_results = []
#     layer_list = [i for i in range(nlayer)]
#     for layer in layer_list:
#         results = []
#         for iteration in [0,1,2,4,8,16,32,64,128,512,1000,2000,5000,10000,20000,50000,100000,110000,120000,130000,140000]:
#             data = torch.load('../cos_results/pythia-{}-{}-rank.pt'.format(model_name, iteration), map_location='cpu')[layer]
#             results.append(data.item())
#         all_layer_results.append(results)

#     iterations = [0,1,2,4,8,16,32,64,128,512,1000,2000,5000,10000,20000,50000,100000,110000,120000,130000,140000]
#     k_iterations = [item/1000 for item in iterations]

#     cmaps = plt.get_cmap('rainbow', nlayer)
#     for i in range(len(layer_list)):
#         plt.plot(k_iterations, all_layer_results[i], label='layer{}'.format(layer_list[i]), color=cmaps(i/nlayer))
#         plt.scatter(k_iterations[0], all_layer_results[i][0], marker='*', color=cmaps(i/nlayer), s=50)
#     plt.ylabel("Cosine Similarity")
#     plt.xlabel("Minibatch (k)")
#     # plt.title('Pythia-1.4b, Layers: 24, val_loss: 2.625')
#     norm_iters = plt.Normalize(0, nlayer - 1)
#     sm = cm.ScalarMappable(cmap=cmaps, norm=norm_iters)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=plt.gca())
#     cbar.set_label('', rotation=270, labelpad=15)
#     plt.savefig('{}_cos-weight.svg'.format(model_name), bbox_inches='tight')
#     plt.close()


# layer_list = [6,12,24,16,24,32]

# for model_name, nlayer in zip(['70m', '160m', '410m', '1b', '1.4b', '2.8b'], layer_list):
#     plt.figure(figsize=(5,3.5))
#     all_layer_results = []
#     layer_list = [i for i in range(nlayer)]
#     for layer in layer_list:
#         results = []
#         for iteration in [0,1,2,4,8,16,32,64,128]:
#             data = torch.load('../cos_results/pythia-{}-{}-rank.pt'.format(model_name, iteration), map_location='cpu')[layer]
#             results.append(data.item())
#         all_layer_results.append(results)

#     iterations = [0,1,2,4,8,16,32,64,128]
#     k_iterations = [item/1 for item in iterations]

#     cmaps = plt.get_cmap('rainbow', nlayer)
#     for i in range(len(layer_list)):
#         plt.plot(k_iterations, all_layer_results[i], label='layer{}'.format(layer_list[i]), color=cmaps(i/nlayer))
#         plt.scatter(k_iterations[0], all_layer_results[i][0], marker='*', color=cmaps(i/nlayer), s=50)
#     plt.ylabel("Cosine Similarity")
#     plt.xlabel("Minibatch")
#     # plt.title('Pythia-1.4b, Layers: 24, val_loss: 2.625')
#     norm_iters = plt.Normalize(0, nlayer - 1)
#     sm = cm.ScalarMappable(cmap=cmaps, norm=norm_iters)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=plt.gca())
#     cbar.set_label('', rotation=270, labelpad=15)
#     plt.savefig('{}_cos-weight-sup.svg'.format(model_name), bbox_inches='tight')
#     plt.close()









opt = torch.load('../cos_results/opt-6.7b-cos-weight.pt')
llama = torch.load('../cos_results/llama-2-cos-weight.pt')
vit = torch.load('../cos_results/vit-huge-cos-weight.pt')
bert = torch.load('../cos_results/bert-base-cos-weight.pt')

def mean_results(data):
    meandata = []
    nlayer = len(data.keys())
    for layer in range(nlayer):
        meandata.append(data[layer].detach().numpy())
    return meandata

plt.figure(figsize=(4,3.5))
plt.grid(linestyle='dashed', zorder=0)
opt = mean_results(opt)
llama = mean_results(llama)
vit = mean_results(vit)
bert = mean_results(bert)

plt.plot(bert, color='red', label='BERT-Base')
plt.plot(llama, color='forestgreen', label='OPT-6.7B')
plt.plot(vit, color='darkorange', label='LLAMA-2-7B')
plt.plot(opt, color='royalblue', label='ViT-Huge')
plt.xlabel('Layer Index')
plt.ylabel('Cosine Similarity')
plt.title('Final Checkpoint')
plt.legend()
plt.ylim(0, 0.5)
plt.savefig('other_arch.pdf', bbox_inches='tight')
plt.close()
# plt.show()