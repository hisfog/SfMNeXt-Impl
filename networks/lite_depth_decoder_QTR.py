from __future__ import absolute_import, division, print_function
# from visualizer import get_local
import torch
import torch.nn as nn
from .layers import PixelWiseDotProduct_for_dense, PixelWiseDotProduct_for_summary, FullQueryLayer
class Lite_Depth_Decoder_QueryTr(nn.Module):
    def __init__(self, in_channels, embedding_dim=128, patch_size=16, num_heads=4, query_nums=100, dim_out=256, norm='linear',
                 min_val=0.001, max_val=10) -> None:
        super(Lite_Depth_Decoder_QueryTr, self).__init__()
        self.norm = norm
        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim,
                                           kernel_size=patch_size, stride=patch_size, padding=0)
        self.positional_encodings = nn.Parameter(torch.rand(500, embedding_dim), requires_grad=True)
        # encoder_layers = nn.modules.transformer.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        encoder_layers = nn.modules.transformer.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=512) # for resnet18
        self.transformer_encoder = nn.modules.transformer.TransformerEncoder(encoder_layers, num_layers=4)
        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)

        self.full_query_layer = FullQueryLayer()
        self.bins_regressor = nn.Sequential(nn.Linear(embedding_dim*query_nums, 16*query_nums),
                                       nn.LeakyReLU(),
                                       nn.Linear(16*query_nums, 16*16),
                                       nn.LeakyReLU(),
                                       nn.Linear(16*16, dim_out))

        self.convert_to_prob = nn.Sequential(nn.Conv2d(query_nums, dim_out, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))
        self.query_nums = query_nums

        self.min_val = min_val
        self.max_val = max_val

    # @get_local('x0')
    def forward(self, x0):
        embeddings_0 = self.embedding_convPxP(x0.clone())
        embeddings_0 = embeddings_0.flatten(2)
        embeddings_0 = embeddings_0 + self.positional_encodings[:embeddings_0.shape[2], :].T.unsqueeze(0)
        embeddings_0 = embeddings_0.permute(2, 0, 1)
        total_queries = self.transformer_encoder(embeddings_0)

        x0 = self.conv3x3(x0)
        queries = total_queries[:self.query_nums, ...]
        queries = queries.permute(1, 0, 2)

        energy_maps, summarys = self.full_query_layer(x0, queries)
        bs, Q, E = summarys.shape
        y = self.bins_regressor(summarys.view(bs, Q*E))


        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), energy_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)
        
        out = self.convert_to_prob(energy_maps)
        bin_widths = (self.max_val - self.min_val) * y
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)
        outputs = {}
        outputs["disp", 0] = pred
        # outputs["attn", 0] = out
        # outputs["bins", 0] = bin_edges
        return outputs



