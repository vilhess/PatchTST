import torch

def get_tensors(data):
    (in1, y1), (in2, y2), (in3, y3), (in4, y4), (in5, y5), (in6, y6) = data # 

    # Only concatenate non-None batches
    tensors_in = []
    targets_in = []

    # Add valid tensors to the list
    if in1 is not None and y1 is not None:
        tensors_in.append(in1)
        targets_in.append(y1)
    if in2 is not None and y2 is not None:
        tensors_in.append(in2)
        targets_in.append(y2)
    if in3 is not None and y3 is not None:
        tensors_in.append(in3)
        targets_in.append(y3)
    if in4 is not None and y4 is not None:
        tensors_in.append(in4)
        targets_in.append(y4)
    if in5 is not None and y5 is not None:
        tensors_in.append(in5)
        targets_in.append(y5)
    if in6 is not None and y6 is not None:
        tensors_in.append(in6)
        targets_in.append(y6)

    # If there are valid tensors to concatenate, proceed with concatenation
    if tensors_in:
        all_in = torch.cat(tensors_in, dim=0)
        all_tar = torch.cat(targets_in, dim=0)
    
    return all_in, all_tar