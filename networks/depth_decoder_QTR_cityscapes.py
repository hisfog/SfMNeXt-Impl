from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
from .layers import PixelWiseDotProduct_for_dense, PixelWiseDotProduct_for_summary, FullQueryLayer
class Depth_Decoder_QueryTr_City(nn.Module):
    def __init__(self, in_channels, embedding_dim=128, patch_size=16, num_heads=4, query_nums=100, dim_out=256, norm='linear',
                 min_val=0.001, max_val=10) -> None:
        super(Depth_Decoder_QueryTr_City, self).__init__()
        self.norm = norm
        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim,
                                           kernel_size=patch_size, stride=patch_size, padding=0)
        self.positional_encodings = nn.Parameter(torch.rand(500, embedding_dim), requires_grad=True)
        encoder_layers = nn.modules.transformer.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.modules.transformer.TransformerEncoder(encoder_layers, num_layers=4)  # takes shape S,N,E
        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)

        # self.summary_layer = PixelWiseDotProduct_for_summary()
        # self.dense_layer = PixelWiseDotProduct_for_dense()
        self.full_query_layer = FullQueryLayer()
        self.motion_query_layer = PixelWiseDotProduct_for_dense()
        self.motion_regressor = nn.Sequential(nn.Linear(embedding_dim*10, 10*10),
                                       nn.LeakyReLU(),
                                       # nn.Linear(10*10, 10*10),
                                       # nn.LeakyReLU(),
                                       nn.Linear(10*10, 100))
        self.bins_regressor = nn.Sequential(nn.Linear(embedding_dim*query_nums, 16*query_nums),
                                       nn.LeakyReLU(),
                                       nn.Linear(16*query_nums, 16*16),
                                       nn.LeakyReLU(),
                                       nn.Linear(16*16, dim_out))

        self.convert_to_prob_motion = nn.Sequential(nn.Conv2d(10, 1, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))
        self.convert_to_prob = nn.Sequential(nn.Conv2d(query_nums, dim_out, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))
        self.query_nums = query_nums

        self.min_val = min_val
        self.max_val = max_val
    def forward(self, x0):
        # print(x0.shape, " ==33 x0.shape")
        embeddings_0 = self.embedding_convPxP(x0.clone())  # .shape = n,c,s = n, embedding_dim, s
        # print(embeddings_0.shape, " ==35 embeddings_0.shape")
        embeddings_0 = embeddings_0.flatten(2)
        # print(embeddings_0.shape, " ==37 embeddings_0.shape")
        embeddings_0 = embeddings_0 + self.positional_encodings[:embeddings_0.shape[2], :].T.unsqueeze(0)
        # print(embeddings_0.shape, " ==38 embeddings_0.shape")
        embeddings_0 = embeddings_0.permute(2, 0, 1)
        # print(embeddings_0.shape, "==37 embeddings_0.shape")
        total_queries = self.transformer_encoder(embeddings_0)  # .shape = S, N, E
        # print(total_queries.shape, " ==43 total_queries.shape")

        x0 = self.conv3x3(x0)
        # print(x0.shape, " ==46 x0.shape")
        queries = total_queries[10:self.query_nums, ...]
        queries_motion = total_queries[:10, ...]
        # print(queries.shape, " ==48 queries.shape")
        queries = queries.permute(1, 0, 2)
        queries_motion = queries_motion.permute(1, 0, 2)
        # print(queries.shape, " ==50 queries.shape")

        energy_maps, summarys = self.full_query_layer(x0, queries) # [bs, 128, E]
        energy_maps_motion, _ = self.motion_query_layer(x0, queries) # [bs, 128, E]
        # summarys = self.summary_layer(x0, queries) # [bs, 128, E]
        bs, Q, E = summarys.shape
        # bs_m, Q_m, E_m = summarys_motion.shape
        # print(summarys.shape, " ==52 summarys.shape")
        y = self.bins_regressor(summarys.view(bs, Q*E))
        # y_m = self.motion_regressor(summarys_motion.view(bs_m, Q_m*E))
        # print(y.shape, " ==55 y.shape")

        # energy_maps = self.dense_layer(x0, queries)
        # print(energy_maps.shape, " ==57 energy_maps.shape")

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
        out_m = self.convert_to_prob_motion(energy_maps_motion)
        # print(out.shape, " ==71 out.shape")
        bin_widths = (self.max_val - self.min_val) * y  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        # print(bin_widths.shape, " ==74 bin_widths.shape")
        bin_edges = torch.cumsum(bin_widths, dim=1) # 累积和
        # print(bin_edges.shape, " ==76 bin_edges.shape")

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        #    0, 1, 2, 3, 4, 5 +
        # 0, 1, 2, 3, 4, 5 =
        #   0.5 1.5 ... 4.5
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)
        # print(centers.shape, " ==79 centers.shape")

        # print(out.shape, " ==67")
        # print(centers.shape, " ==68")
        pred = torch.sum(out * centers, dim=1, keepdim=True)
        outputs = {}
        outputs["disp", 0] = pred
        outputs["motion", 0] = out_m
        # outputs["attn", 0] = out
        # outputs["bins", 0] = bin_edges
        # outputs["mean", 0] = bin_edges
        return outputs


