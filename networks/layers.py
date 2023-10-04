import torch
import torch.nn as nn

class FullQueryLayer(nn.Module):
    def __init__(self) -> None:
        super(FullQueryLayer, self).__init__()
    def forward(self, x, K):
        """
        given feature map of size [bs, E, H, W], and queries of size [bs, Q, E]
        return Q energy maps corresponding to Q queries of shape [bs, Q, H, W]
        and add feature noise to x of the same shape as input [bs, E, H, W]
        and summary_embedding of shape [bs, Q, E]
        """
        n, c, h, w = x.size() # bs, E, H, W
        _, cout, ck = K.size() # bs, Q, E
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1))
        y_norm = torch.softmax(y, dim=1)
        summary_embedding = torch.matmul(y_norm.permute(0, 2, 1), x.view(n, c, h*w).permute(0, 2, 1))
        y = y.permute(0, 2, 1).view(n, cout, h, w)
        return y, summary_embedding


class PixelWiseDotProduct_for_summary(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct_for_summary, self).__init__()

    def forward(self, x, K):
        """
        given feature map x of size [bs, E, H, W], and queries of size [bs, Q, E]
        return Q summary_embedding corresponding to Q queries of shape [bs, Q, E]
        """
        n, c, h, w = x.size() # bs, E, H, W
        _, cout, ck = K.size() # bs, Q, E
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1))
        y = torch.softmax(y, dim=1)
        summary_embedding = torch.matmul(y.permute(0, 2, 1), x.view(n, c, h*w).permute(0, 2, 1))
        return summary_embedding  

class regressor_for_short_vector(nn.Module):
    def __init__(self, embedding_dim=128, dim_out=128):
        super(regressor_for_short_vector, self).__init__()
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))
    def forward(self, x):
        res_vector = self.regressor(x)
        return res_vector
        

class PixelWiseDotProduct_for_dense(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct_for_dense, self).__init__()

    def forward(self, x, K):
        """
        given feature map x of size [bs, E, H, W], and queries of size [bs, Q, E]
        return Q energy maps corresponding to Q queries of shape [bs, Q, H, W]
        """
        n, c, h, w = x.size() # bs, E, H, W
        _, cout, ck = K.size() # bs, Q, E
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1))
        return y.permute(0, 2, 1).view(n, cout, h, w)


