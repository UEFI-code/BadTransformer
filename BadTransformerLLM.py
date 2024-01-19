import torch
import torch.nn as nn
class myBadTransfomerBlock(nn.Module):
    def __init__(self, dim=4096, activation=nn.ReLU(), debug=False):
        super().__init__()
        self.li1 = nn.Linear(dim, dim, bias=False)
        self.li2 = nn.Linear(dim, dim, bias=False)
        self.li3 = nn.Linear(dim, dim, bias=False)
        self.li4 = nn.Linear(dim, dim, bias=False)
        self.activation = activation
        self.debug = debug

    def forward(self, x):
        xA, xB, xC = self.li1(x), self.li2(x), self.li3(x)
        xA, xB, xC = self.activation(xA), self.activation(xB), self.activation(xC)
        xSqure = torch.matmul(xA.transpose(1, 2), xB) # Here is to semantic hybrid. Expected shape: (batch, dim, dim)
        if self.debug:
            print(xSqure.shape)
        xO = torch.matmul(xC, xSqure) # Here is to re-mapping the meaning of each token embedding
        return self.li4(xO)

class myBadTransformerUnit(nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()
        self.badtrans = []
        for _ in range(num_layers):
            self.badtrans.append(myBadTransfomerBlock())
        self.badtrans = nn.ModuleList(self.badtrans)

    def forward(self, x):
        for badtrans in self.badtrans:
            x = badtrans(x)
        return x

if __name__ == "__main__":
    x = torch.randn(1, 16, 4096)
    badBlock = myBadTransfomerBlock(debug=True)
    print(badBlock(x))