import torch
import matplotlib.cm as cm
import matplotlib.pyplot as plt

a_norm = torch.load('../opt_6.7b_wiki103_emb-a-norm.pt', map_location=torch.device('cpu'), weights_only=False)
v_norm = torch.load('../opt_6.7b_wiki103_emb-v-norm.pt', map_location=torch.device('cpu'), weights_only=False)

# # define subplot grid
a = 4  # number of rows
b = 8  # number of columns
c = 1  # initialize plot counter
#
fig = plt.figure(figsize = (4,8))
plt.suptitle("32 layer OPT", fontsize = 18)
#
for i in range(32):
    plt.subplot(a, b, c)
    plt.title('layer: {}'.format(i))
    plt.xlabel(i)

    # plt.plot(v_norm[i], color='blue')
    c = c + 1
    print(a_norm[i])
    print(v_norm[i])
    plt.plot(a_norm[i] - v_norm[i], color='green')

plt.show()
#
# print(len(a_norm[0]))

