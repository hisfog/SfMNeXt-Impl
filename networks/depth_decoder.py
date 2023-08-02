# pyright: reportGeneralTypeIssues=false
import torch
import torch.nn as nn

from .layers import PatchTransformerEncoder, PixelWiseDotProduct
DIMS_EMBEDDING = 54
NUM_HEADS = 6

class DepthDecoder(nn.Module):
    def __init__(self, in_channels, n_query_channels=DIMS_EMBEDDING, patch_size=16, dim_out=256,
                 embedding_dim=DIMS_EMBEDDING, num_heads=NUM_HEADS, norm='linear'):
        super(DepthDecoder, self).__init__()
        self.norm = norm
        self.n_query_channels = n_query_channels
        self.patch_transformer = PatchTransformerEncoder(in_channels, patch_size, embedding_dim, num_heads)
        self.dot_product_layer = PixelWiseDotProduct()

        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))
                                       # 100是nbins
        self.conv_out = nn.Sequential(nn.Conv2d(DIMS_EMBEDDING, dim_out, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))

        self.num_classes = 100 
        # self.min_val = 0.001
        # self.min_val = 0.0
        self.min_val = 0.1
        self.max_val = 10.0
        # self.max_val = 10
    def forward(self, x):
        # n, c, h, w = x.size()
        # print(x.shape)

        self.outputs = {}
        # self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
        tgt = self.patch_transformer(x.clone())  # .shape = S, N, E
        # print('tgt.shape = ', tgt.shape)#[242,2,DIMS_EMBEDDING]

        x = self.conv3x3(x)
        # print('conv3x3 x= ', x.shape)#[2,DIMS_EMBEDDING,176,352]
        regression_head, queries = tgt[0, ...], tgt[1:self.n_query_channels + 1, ...]

        # Change from S, N, E to N, S, E
        queries = queries.permute(1, 0, 2)
        range_attention_maps = self.dot_product_layer(x, queries)  # .shape = n, n_query_channels, h, w
        # print(range_attention_maps.shape, " 44")
        # print('range_attention_maps = ', range_attention_maps.shape)
        # range_attention_maps = [2,DIMS_EMBEDDING,176,352]

        y = self.regressor(regression_head)  # .shape = N, dim_out
        # print('regressor out = ', y.shape)#[2,100]
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)
        # print('normed y = ', y.shape)#[2,100]
        # self.outputs[("disp", i)] = y
        # return y, range_attention_maps
        # print(range_attention_maps.shape)
        out = self.conv_out(range_attention_maps)
        bin_widths = (self.max_val - self.min_val) * y  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)
        self.outputs["disp", 0] = pred
        return self.outputs
