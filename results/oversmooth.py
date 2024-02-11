import torch 
import numpy as np 
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt 


# opt = torch.load('opt_6.7b_wiki103_emb-emb_before-abs.pt')
# llama = torch.load('llama_2_7b_wiki103_emb-emb_before-abs.pt')
# vit = torch.load('beans_obs_emb_vit_huge-emb_before-abs.pt')
# bert = torch.load('bert_base_wiki103_step_2000000-emb_before-abs.pt')



# # opt = torch.load('opt_6.7b_wiki103_emb-emb_after-abs.pt')
# # llama = torch.load('llama_2_7b_wiki103_emb-emb_after-abs.pt')
# # vit = torch.load('beans_obs_emb_vit_huge-emb_after-abs.pt')
# # bert = torch.load('bert_base_wiki103_step_2000000-emb_after-abs.pt')



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
# # plt.plot(opt, color='royalblue', label='ViT-Huge')
# plt.xlabel('Layer Index')
# plt.ylabel('Cosine Similarity')
# plt.title('Final Checkpoint')
# plt.legend()
# # plt.savefig('other_arch.svg', bbox_inches='tight')
# # plt.close()
# plt.show()










# opt = torch.load('opt_6.7b_wiki103_emb-emb_before-abs.pt')
# llama = torch.load('llama_2_7b_wiki103_emb-emb_before-abs.pt')
# vit = torch.load('beans_obs_emb_vit_huge-emb_before-abs.pt')
# bert = torch.load('bert_base_wiki103_step_2000000-emb_before-abs.pt')

# opt = torch.load('opt_6.7b_wiki103_emb-emb_after-abs.pt')
# llama = torch.load('llama_2_7b_wiki103_emb-emb_after-abs.pt')
# vit = torch.load('beans_obs_emb_vit_huge-emb_after-abs.pt')
# bert = torch.load('bert_base_wiki103_step_2000000-emb_after-abs.pt')


llama = torch.load('llama_2_7b_wiki103_emb-emb_before-abs.pt')
llama_after = torch.load('llama_2_7b_wiki103_emb-emb_after-abs.pt')


llama_position = torch.load('llama_2_7b_wiki103_emb_position-emb_before-abs.pt')
llama_position_after = torch.load('llama_2_7b_wiki103_emb_position-emb_after-abs.pt')



llama_position_1 = torch.load('llama_2_7b_wiki103_emb_position_0.5_1-emb_before-abs.pt')
llama_position_after_1 = torch.load('llama_2_7b_wiki103_emb_position_0.5_1-emb_after-abs.pt')

llama_position_2 = torch.load('llama_2_7b_wiki103_emb_position_0.5_2-emb_before-abs.pt')
llama_position_after_2 = torch.load('llama_2_7b_wiki103_emb_position_0.5_2-emb_after-abs.pt')

llama_position_3 = torch.load('llama_2_7b_wiki103_emb_position_0.5_3-emb_before-abs.pt')
llama_position_after_3 = torch.load('llama_2_7b_wiki103_emb_position_0.5_3-emb_after-abs.pt')

llama_position_4 = torch.load('llama_2_7b_wiki103_emb_position_0.8_1-emb_before-abs.pt')
llama_position_after_4 = torch.load('llama_2_7b_wiki103_emb_position_0.8_1-emb_after-abs.pt')

llama_position_5 = torch.load('llama_2_7b_wiki103_emb_position_0.8_1-emb_before-abs.pt')
llama_position_after_5 = torch.load('llama_2_7b_wiki103_emb_position_0.8_1-emb_after-abs.pt')

llama_position_6 = torch.load('llama_2_7b_wiki103_emb_position_0.8_1-emb_before-abs.pt')
llama_position_after_6 = torch.load('llama_2_7b_wiki103_emb_position_0.8_1-emb_after-abs.pt')

llama_position_7 = torch.load('llama_2_7b_wiki103_emb_position_1_1-emb_before-abs.pt')
llama_position_after_7 = torch.load('llama_2_7b_wiki103_emb_position_1_1-emb_after-abs.pt')

llama_position_8 = torch.load('llama_2_7b_wiki103_emb_position_1_2-emb_before-abs.pt')
llama_position_after_8 = torch.load('llama_2_7b_wiki103_emb_position_1_2-emb_after-abs.pt')

llama_position_9 = torch.load('llama_2_7b_wiki103_emb_position_1_3-emb_before-abs.pt')
llama_position_after_9 = torch.load('llama_2_7b_wiki103_emb_position_1_3-emb_after-abs.pt')





# llama_position_1 = torch.load('llama_2_7b_wiki103_emb_position_0.8_1-emb_before-abs.pt')
# llama_position_after_1 = torch.load('llama_2_7b_wiki103_emb_position_0.8_1-emb_after-abs.pt')

# llama_position_2 = torch.load('llama_2_7b_wiki103_emb_position_0.8_1.1-emb_before-abs.pt')
# llama_position_after_2 = torch.load('llama_2_7b_wiki103_emb_position_0.8_1.1-emb_after-abs.pt')

# llama_position_3 = torch.load('llama_2_7b_wiki103_emb_position_0.8_1.2-emb_before-abs.pt')
# llama_position_after_3 = torch.load('llama_2_7b_wiki103_emb_position_0.8_1.2-emb_after-abs.pt')

# llama_position_4 = torch.load('llama_2_7b_wiki103_emb_position_0.8_1.05-emb_before-abs.pt')
# llama_position_after_4 = torch.load('llama_2_7b_wiki103_emb_position_0.8_1.05-emb_after-abs.pt')

# llama_position_5 = torch.load('llama_2_7b_wiki103_emb_position_0.8_1.3-emb_before-abs.pt')
# llama_position_after_5 = torch.load('llama_2_7b_wiki103_emb_position_0.8_1.3-emb_after-abs.pt')

# llama_position_6 = torch.load('llama_2_7b_wiki103_emb_position_0.9_1-emb_before-abs.pt')
# llama_position_after_6 = torch.load('llama_2_7b_wiki103_emb_position_0.9_1-emb_after-abs.pt')

# llama_position_7 = torch.load('llama_2_7b_wiki103_emb_position_0.9_1.05-emb_before-abs.pt')
# llama_position_after_7 = torch.load('llama_2_7b_wiki103_emb_position_0.9_1.05-emb_after-abs.pt')

# llama_position_8 = torch.load('llama_2_7b_wiki103_emb_position_0.9_1.1-emb_before-abs.pt')
# llama_position_after_8 = torch.load('llama_2_7b_wiki103_emb_position_0.9_1.1-emb_after-abs.pt')

# llama_position_9 = torch.load('llama_2_7b_wiki103_emb_position_0.9_1.2-emb_before-abs.pt')
# llama_position_after_9 = torch.load('llama_2_7b_wiki103_emb_position_0.9_1.2-emb_after-abs.pt')




# llama_position_after_1 = torch.load('llama_2_7b_wiki103_emb_position_0.1_0.1-emb_after-abs.pt')
# llama_position_after_2 = torch.load('llama_2_7b_wiki103_emb_position_0.1_0.3-emb_after-abs.pt')
# llama_position_after_3 = torch.load('llama_2_7b_wiki103_emb_position_0.1_0.5-emb_after-abs.pt')
# llama_position_after_4 = torch.load('llama_2_7b_wiki103_emb_position_0.1_0.7-emb_after-abs.pt')
# llama_position_after_5 = torch.load('llama_2_7b_wiki103_emb_position_0.1_0.9-emb_after-abs.pt')



# llama_position_after_1 = torch.load('llama_2_7b_wiki103_emb_position_0.3_0.1-emb_after-abs.pt')
# llama_position_after_2 = torch.load('llama_2_7b_wiki103_emb_position_0.3_0.3-emb_after-abs.pt')
# llama_position_after_3 = torch.load('llama_2_7b_wiki103_emb_position_0.3_0.5-emb_after-abs.pt')
# llama_position_after_4 = torch.load('llama_2_7b_wiki103_emb_position_0.3_0.7-emb_after-abs.pt')
# llama_position_after_5 = torch.load('llama_2_7b_wiki103_emb_position_0.3_0.9-emb_after-abs.pt')


# llama_position_after_1 = torch.load('llama_2_7b_wiki103_emb_position_0.5_0.1-emb_after-abs.pt')
# llama_position_after_2 = torch.load('llama_2_7b_wiki103_emb_position_0.5_0.3-emb_after-abs.pt')
# llama_position_after_3 = torch.load('llama_2_7b_wiki103_emb_position_0.5_0.5-emb_after-abs.pt')
# llama_position_after_4 = torch.load('llama_2_7b_wiki103_emb_position_0.5_0.7-emb_after-abs.pt')
# llama_position_after_5 = torch.load('llama_2_7b_wiki103_emb_position_0.5_0.9-emb_after-abs.pt')

# llama_position_after_1 = torch.load('llama_2_7b_wiki103_emb_position_0.7_0.1-emb_after-abs.pt')
# llama_position_after_2 = torch.load('llama_2_7b_wiki103_emb_position_0.7_0.3-emb_after-abs.pt')
# llama_position_after_3 = torch.load('llama_2_7b_wiki103_emb_position_0.7_0.5-emb_after-abs.pt')
# llama_position_after_4 = torch.load('llama_2_7b_wiki103_emb_position_0.7_0.7-emb_after-abs.pt')
# llama_position_after_5 = torch.load('llama_2_7b_wiki103_emb_position_0.5_0.9-emb_after-abs.pt')


llama_position_after_1 = torch.load('llama_2_7b_wiki103_emb_temperature_0.5_1-emb_before-abs.pt')
llama_position_after_2 = torch.load('llama_2_7b_wiki103_emb_temperature_1_1-emb_before-abs.pt')
llama_position_after_3 = torch.load('llama_2_7b_wiki103_emb_temperature_2_1-emb_before-abs.pt')
llama_position_after_4 = torch.load('llama_2_7b_wiki103_emb_temperature_3_1-emb_before-abs.pt')
llama_position_after_5 = torch.load('llama_2_7b_wiki103_emb_temperature_3_3-emb_before-abs.pt')

print(llama_position_after_1)
def mean_results(data):
    meandata = []
    nlayer = len(data.keys())
    for layer in range(nlayer):
        meandata.append(np.mean(data[layer]))
    return meandata

plt.figure(figsize=(4,3.5))
plt.grid(linestyle='dashed', zorder=0)
# opt = mean_results(opt)
# llama = mean_results(llama)
# vit = mean_results(vit)
# bert = mean_results(bert)


# llama = mean_results(llama)
# llama_position = mean_results(llama_position)
# plt.plot(llama, color='forestgreen', label='LLAMA-2-7B')
# plt.plot(llama_position, color='royalblue', label='LLAMA-2-7B w. Position Interplation')



llama_after = mean_results(llama_after)
llama_position_after = mean_results(llama_position_after)
llama_position_after_1 = mean_results(llama_position_after_1)
llama_position_after_2 = mean_results(llama_position_after_2)
llama_position_after_3 = mean_results(llama_position_after_3)
llama_position_after_4 = mean_results(llama_position_after_4)
llama_position_after_5 = mean_results(llama_position_after_5)

print(llama_position_after_1)


# llama_position_after_6 = mean_results(llama_position_after_6)
# llama_position_after_7 = mean_results(llama_position_after_7)
# llama_position_after_8 = mean_results(llama_position_after_8)
# llama_position_after_9 = mean_results(llama_position_after_9)


# plt.plot(llama_after, color='forestgreen', label='LLAMA-2-7B after', linestyle='dashdot')
# # plt.plot(llama_position_after, color='royalblue', label='LLAMA-2-7B after w. Position Interplation')

plt.plot(llama_position_after_1, label='LLAMA-2-7B after w. Position Interplation 1')
plt.plot(llama_position_after_2, label='LLAMA-2-7B after w. Position Interplation 2')
plt.plot(llama_position_after_3, label='LLAMA-2-7B after w. Position Interplation 3')
plt.plot(llama_position_after_4, label='LLAMA-2-7B after w. Position Interplation 4')
plt.plot(llama_position_after_5, label='LLAMA-2-7B after w. Position Interplation 5')
# plt.plot(llama_position_after_6, label='LLAMA-2-7B after w. Position Interplation 6')
# plt.plot(llama_position_after_7, label='LLAMA-2-7B after w. Position Interplation 7')
# plt.plot(llama_position_after_8, label='LLAMA-2-7B after w. Position Interplation 8')
# plt.plot(llama_position_after_9, label='LLAMA-2-7B after w. Position Interplation 9')



# plt.plot(opt, color='royalblue', label='ViT-Huge')
plt.xlabel('Layer Index')
plt.ylabel('Cosine Similarity')
plt.title('Final Checkpoint')
plt.legend()
# plt.savefig('other_arch.svg', bbox_inches='tight')
# plt.close()
plt.show()