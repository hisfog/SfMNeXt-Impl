# from prettytable import PrettyTable
from options import MonodepthOptions
# import datasets
import networks

def count_parameters(model):
    # table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        # table.add_row([name, param])
        total_params+=param
    # print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

    
encoder = networks.BaseEncoder.build(model_dim=32)
# depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)
depth_decoder = networks.Depth_Decoder_QueryTr(in_channels=32, patch_size=20,dim_out=128, embedding_dim=32, 
                                                                query_nums=128, num_heads=4,
                                                                min_val=0.001, max_val=10.0)
count_parameters(encoder)
count_parameters(depth_decoder)
