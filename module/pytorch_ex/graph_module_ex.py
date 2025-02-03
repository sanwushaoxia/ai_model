import torch, dgl, time

def graph_2_adjacencyMatrix(graph, norm='both'):
    src, dst = graph.edges()
    num_nodes = graph.num_nodes()
    indices = torch.stack((src, dst), 0)
    values = torch.ones(src.size(0)).to(indices.device)
    A = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

    if norm == 'none':
        return A
    else:
        d = torch.sum(A.to_dense(), 1)
        if norm == 'both':
            d = d ** (-0.5)
            A = torch.einsum('i, ij, j -> ij', d, A, d)
        else:
            d = d ** (-1)
            if norm == 'right':
                A = torch.einsum('i, ij -> ij', d, A)
            elif norm == 'left':
                A = torch.einsum('ij, j -> ij', A, d)
        return A

class GraphConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, weight=True, bias=True, activation=None):
        super(GraphConv, self).__init__()
        if weight:
            self.W = torch.nn.Parameter(torch.rand(in_channels, out_channels) / 10)
        else:
            assert in_channels == out_channels
            self.W = torch.eye(in_channels)

        if bias:
            self.b = torch.nn.Parameter(torch.rand(out_channels) / 10)
        else:
            self.b = torch.zeros(out_channels)

        self.activation = activation
    def forward(self, A, X):
        X = A @ X @ self.W + self.b

        if self.activation is None:
            return X
        else:
            return self.activation(X)

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.gc0 = GraphConv(1433, 140, activation=torch.nn.ReLU(inplace=True))
        self.gc1 = GraphConv(140, 7, activation=torch.nn.ReLU(inplace=True))
    def forward(self, A, X):
        X = self.gc0(A, X)
        X = self.gc1(A, X)
        return X

class GCN_(torch.nn.Module):
    def __init__(self):
        super(GCN_, self).__init__()
        self.gc0 = dgl.nn.pytorch.conv.GraphConv(1433, 140, activation=torch.nn.ReLU(inplace=True))
        self.gc1 = dgl.nn.pytorch.conv.GraphConv(140, 7, activation=torch.nn.ReLU(inplace=True))
    def forward(self, graph, X):
        X = self.gc0(graph, X)
        X = self.gc1(graph, X)
        return X

class GATConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=1, negative_slope=0.2, activation=None, bias=True):
        super(GATConv, self).__init__()
        self.out_channels = out_channels
        self.W = torch.nn.Parameter(torch.rand(in_channels, out_channels, num_heads) / 10)
        self.attn = torch.nn.Parameter(torch.rand(out_channels * 2, num_heads) / 10)
        self.leakyrelu = torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        if bias:
            self.b = torch.nn.Parameter(torch.rand(num_heads, out_channels) / 10)
        else:
            self.b = torch.zeros(num_heads, out_channels)

        self.activation = activation
    def forward(self, A, X):
        X = torch.einsum('ik, kjl -> ijl', X, self.W) # 2708, 7, 2
        X0 = torch.einsum('ijk, jk -> ik', X, self.attn[:self.out_channels, :])
        X1 = torch.einsum('ijk, jk -> ik', X, self.attn[self.out_channels:, :])

        X0 = X0.view(X0.size(0), -1, X0.size(1))
        X1 = X1.view(-1, X1.size(0), X1.size(1))
        X_ = X0 + X1

        X_ = self.leakyrelu(X_)
        X_ = torch.exp(X_)
        A = torch.einsum('ij, ijk -> ijk', A, X_)

        d = torch.sum(A, 1) + 1e-3
        d = d ** (-1)
        A = torch.einsum('ik, ijk -> ijk', d, A)

        X = torch.einsum('ikl, kjl -> ilj', A, X) + self.b
        if self.activation is None:
            return X
        else:
            return self.activation(X)

class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.gat0 = GATConv(1433, 140, activation=torch.nn.ReLU(inplace=True))
        self.gat1 = GATConv(140, 7, activation=torch.nn.ReLU(inplace=True))
    def forward(self, A, X):
        X = self.gat0(A, X)
        X = X.view(X.size(0), -1)
        X = self.gat1(A, X)
        X = X.view(X.size(0), -1)
        return X

class GAT_(torch.nn.Module):
    def __init__(self):
        super(GAT_, self).__init__()
        self.gat0 = dgl.nn.pytorch.conv.GATConv(1433, 140, 1, activation=torch.nn.ReLU(inplace=True))
        self.gat1 = dgl.nn.pytorch.conv.GATConv(140, 7, 1, activation=torch.nn.ReLU(inplace=True))
    def forward(self, A, X):
        X = self.gat0(A, X)
        X = X.view(X.size(0), -1)
        X = self.gat1(A, X)
        X = X.view(X.size(0), -1)
        return X

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = dgl.data.CoraGraphDataset()
    graph = dataset[0].to(device)

    train_mask = graph.ndata['train_mask'].to(device)
    val_mask = graph.ndata['val_mask'].to(device)
    test_mask = graph.ndata['test_mask'].to(device)

    X = graph.ndata['feat'].to(device)
    Y = graph.ndata['label'].to(device)

    A = graph_2_adjacencyMatrix(graph, 'none')
    A = torch.eye(A.size(0)).to(device) + A

    model = GAT().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    correct_val_best = -1

    epochs = 10000

    t0 = time.perf_counter()
    for i in range(1, epochs + 1):
        X_ = model(A, X)
        #X_ = model(graph, X)
        loss = loss_fn(X_[train_mask], Y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, Y_ = torch.max(X_.data, 1)
        correct_train = torch.sum(Y_[train_mask] == Y.data[train_mask])
        correct_val = torch.sum(Y_[val_mask] == Y.data[val_mask])
        correct_test = torch.sum(Y_[test_mask] == Y.data[test_mask])
        if correct_val > correct_val_best:
            correct_val_best = correct_val
            correct_test_best = correct_test
        print('{} train loss={:.2f}, train accuracy={:.2f}, val accuracy={:.2f}, test accuracy={:.2f}'.format(i, loss.item(), correct_train / 140, correct_val / 500, correct_test / 1000))
    t1 = time.perf_counter()
    print('val accuracy={:.2f}, test accuracy={:.2f}, time={}'.format(correct_val_best / 500, correct_test_best / 1000, t1 - t0))
