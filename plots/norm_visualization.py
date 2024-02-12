import torch

a_norm = torch.load('../opt_6.7b_wiki103_emb-a-norm.pt', map_location=torch.device('cpu'), weights_only=False)
v_norm = torch.load('../opt_6.7b_wiki103_emb-v-norm.pt', map_location=torch.device('cpu'), weights_only=False)


print(len(a_norm[0]))