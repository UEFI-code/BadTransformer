import torch
import torch.nn as nn
class myBadTransfomerBlock(nn.Module):
    def __init__(self, dim=4096, deepth = 2, activation=nn.ReLU(), debug=False):
        super().__init__()
        self.encodingGroupA = nn.Sequential()
        self.encodingGroupB = nn.Sequential()
        self.encodingGroupC = nn.Sequential()
        self.decodingGroup = nn.Sequential()
        for _ in range(deepth):
            self.encodingGroupA.append(nn.Linear(dim, dim, bias=False))
            self.encodingGroupA.append(activation)
            self.encodingGroupB.append(nn.Linear(dim, dim, bias=False))
            self.encodingGroupB.append(activation)
            self.encodingGroupC.append(nn.Linear(dim, dim, bias=False))
            self.encodingGroupC.append(activation)
        self.debug = debug
        self.dim = dim

    def forward(self, x):
        xA, xB, xC = self.encodingGroupA(x), self.encodingGroupB(x), self.encodingGroupC(x)
        xSqure = torch.matmul(xA.transpose(1, 2), xB) # Here is to semantic hybrid. Expected shape: (batch, dim, dim)
        if xSqure.shape[1] != self.dim or xSqure.shape[2] != self.dim:
            raise ValueError(f'The shape of xSqure is not expected. Expected: (batch, {self.dim}, {self.dim}), Got: {xSqure.shape}. Maybe you should check the dimension of input.')
        if self.debug:
            print(f'Debug: xSqure shape: {xSqure.shape}')
        xO = torch.matmul(xC, xSqure) # Here is to re-mapping the meaning of each token embedding
        return self.decodingGroup(xO)

class myBadTransformerUnit(nn.Module):
    def __init__(self, dim = 4096, encodingDeepth = 2, num_layers=2, debug=False):
        super().__init__()
        self.badtrans = nn.Sequential()
        for _ in range(num_layers):
            self.badtrans.append(myBadTransfomerBlock(dim=dim, deepth=encodingDeepth, debug=debug))

    def forward(self, x):
        return self.badtrans(x)

if __name__ == "__main__":
    x = torch.randn(1, 16, 4096)
    badBlock = myBadTransfomerBlock(debug=True)
    print(badBlock(x))
    badBlocks = myBadTransformerUnit(debug=True)
    print(badBlocks(x))