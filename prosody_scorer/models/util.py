import torch


def mean_pooling(feature_tensor, mask):
    mean = torch.sum(feature_tensor * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    return mean


def create_mask(feature_embedding, seq_lengths):
    device = feature_embedding.device
    B, T, D = feature_embedding.shape
    range_tensor_for_mask = torch.arange(T).expand(B, T).to(device)
    mask = range_tensor_for_mask < seq_lengths.unsqueeze(1)
    mask = mask.unsqueeze(2).expand(B, T, D)

    return mask
