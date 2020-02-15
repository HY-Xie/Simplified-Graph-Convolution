# networkx是用Python语言开发的图论与复杂网络建模工具，
# 其中karate_club_graph是跆拳道俱乐部数据，to_numpy_matrix用于计算图的邻接矩阵
from networkx import karate_club_graph, to_numpy_matrix

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use("Agg")


import torch 
from torch import nn, optim
import torch.nn.functional as F 

np.random.seed(666)
torch.random.manual_seed(666)

# 用于获取跆拳道俱乐部数据，并返回邻接矩阵和标签
def get_graph_related():
    G = karate_club_graph()
    nodes = sorted(list(G.nodes()))
    A = to_numpy_matrix(G)  # np.float64
    labels = [[1,0] if G.nodes[i]['club']=='Officer' else [0,1] for i in nodes]  # list (int)
    label_dict = {0:'Officer', 1:'Mr. Hi'}
    X = np.random.randn(A.shape[0], 2)
    return A, X, labels, label_dict 

def plot_original_data(X, labels):
    
    color = []
    for label in labels:
        if label[0] == 1:
            color.append('r')
        else:
            color.append('b')
    fig, ax = plt.subplots(1,1)
    ax.scatter(X[:,0], X[:,1], c=color)
    
    plt.show()

class GraphConv(nn.Module):
    def __init__(self, A, labels, X):
        super(GraphConv, self).__init__()
        labels = np.array(labels)
        self.labels = torch.from_numpy(labels).float().requires_grad_(False)
        X = torch.from_numpy(X).float().requires_grad_(True)
        self.X = torch.nn.Parameter(X)

        I = np.eye(A.shape[0])
        AI = A + I
        self.AI = torch.from_numpy(AI).float().requires_grad_(False)
        self.D = torch.diag(torch.sum(self.AI, axis=0)**-0.5)

        W = torch.randn(2, 2).requires_grad_(True)
        self.W = torch.nn.Parameter(W)

        self.register_parameter(name='learned_X', param=self.X)
        self.register_parameter(name='learned_W', param=self.W)
        
        # print(self.W.requires_grad)
    
    def forward(self):
        out = self.D @ self.AI @ self.D @ self.X @ self.W
        out = F.softmax(F.relu(out, inplace=True))
        # out = F.softmax(out)
        
        return out, self.X

def train(A, labels, X):
    model = GraphConv(A, labels, X)
    print(sum(p.numel() for p in model.parameters()))
    model.train()
    print(model.parameters())
    
    # return
    optimizer = optim.Adam(model.parameters(), lr=0.8)
    
    total_learned_X = []
    for i in range(20):
        out, learn_X = model()
        
        # print(out.shape, model.labels.shape)
        # print(out)
        # break
        loss = F.binary_cross_entropy(out, model.labels)
        print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_learned_X.append(model.X.detach().numpy())
    # return 
    final_X = model.X.detach().numpy()
    final_W = model.W

    return final_X, final_W, total_learned_X


if __name__ == '__main__':
    A, X, labels, label_dict = get_graph_related()
    plot_original_data(X, labels)


    learned_X, learned_W, total_learned_X = train(A, labels, X)
    plot_original_data(learned_X, labels)


    
    
    