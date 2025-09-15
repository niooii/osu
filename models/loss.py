# implementation of custom loss funcitons

import torch

def avg_spinner_speed(
        map_features,
        # (B, T, 2) tensor: positions normalized
        pos,                  
        # index of the is_spinner feature, in case i changei t later
        feat_idx_is_spinner=4,
        eps=1e-8,
        # if True, returns CPU floats (per-sample list and scalar); 
        # if False, returns tensors.
        detach=True):

    if not torch.is_tensor(pos):
        pos = torch.as_tensor(pos, dtype=torch.float32)
    if not torch.is_tensor(map_features):
        map_features = torch.as_tensor(map_features, dtype=torch.float32)

    device = pos.device
    B, T, _ = pos.shape

    spinner_feat = map_features[..., feat_idx_is_spinner].to(device)
    spinner_mask = (spinner_feat > 0.5).float()                 # (B, T)

    vel = pos[:, 1:, :] - pos[:, :-1, :]                        # (B, T-1, 2)
    speed = torch.sqrt((vel ** 2).sum(dim=-1) + eps)            # (B, T-1)
    mask_v = spinner_mask[:, 1:]                                # (B, T-1)
    counts = mask_v.sum(dim=1)                                  # (B,)

    # per-sample numerator and safe denom
    numer = (speed * mask_v).sum(dim=1)                         # (B,)
    denom_safe = counts.clamp(min=1.0)                          # (B,)
    per_sample_mean = numer / denom_safe                        # (B,)
    # zero out samples that had zero spinner frames
    per_sample_mean = per_sample_mean * (counts > 0).float()

    # batch mean over all spinner-aligned frames
    total_numer = numer.sum()
    total_denom = counts.sum().clamp(min=1.0)
    batch_mean = total_numer / total_denom                      # scalar

    if detach:
        return per_sample_mean.detach().cpu().numpy(), float(batch_mean.detach().cpu().item()), counts.detach().cpu().numpy()
    else:
        return per_sample_mean, batch_mean, counts


def spinner_loss(real_pos, fake_pos, map_features, feat_idx_is_spinner=4, eps=1e-8):
    """
    Penalizes fake spinning velocity being less than real spinning velocity.
    zero loss if fake velocity >= real velocity.
    """
    if not torch.is_tensor(real_pos):
        real_pos = torch.as_tensor(real_pos, dtype=torch.float32)
    if not torch.is_tensor(fake_pos):
        fake_pos = torch.as_tensor(fake_pos, dtype=torch.float32)
    if not torch.is_tensor(map_features):
        map_features = torch.as_tensor(map_features, dtype=torch.float32)

    _, real_spinner_speed, real_counts = avg_spinner_speed(
        map_features, real_pos, feat_idx_is_spinner, eps, detach=False
    )
    _, fake_spinner_speed, fake_counts = avg_spinner_speed(
        map_features, fake_pos, feat_idx_is_spinner, eps, detach=False
    )
    
    velocity_deficit = torch.clamp(real_spinner_speed - fake_spinner_speed, min=0.0)

    # print(f"{real_spinner_speed} - {fake_spinner_speed} = {velocity_deficit}");
    
    return velocity_deficit


def spinner_mse_loss(real_pos, fake_pos, map_features, feat_idx_is_spinner=4, eps=1e-8):
    """
    MSE between real and fake positions for spinner frames only.
    """
    if not torch.is_tensor(real_pos):
        real_pos = torch.as_tensor(real_pos, dtype=torch.float32)
    if not torch.is_tensor(fake_pos):
        fake_pos = torch.as_tensor(fake_pos, dtype=torch.float32)
    if not torch.is_tensor(map_features):
        map_features = torch.as_tensor(map_features, dtype=torch.float32)

    device = real_pos.device
    fake_pos = fake_pos.to(device)
    map_features = map_features.to(device)

    # mask over time steps where spinner is active
    spinner_feat = map_features[..., feat_idx_is_spinner]
    spinner_mask = (spinner_feat > 0.5).float()  # (B, T)

    # squared error per coordinate
    diff = real_pos - fake_pos                   # (B, T, 2)
    se = diff.pow(2)                             # (B, T, 2)

    # apply mask (broadcast across x/y)
    masked_se = se * spinner_mask.unsqueeze(-1)  # (B, T, 2)

    # total squared error over all spinner-aligned coordinates
    numer = masked_se.sum()
    # count of spinner-aligned coordinates
    denom = (spinner_mask.sum() * real_pos.shape[-1]).clamp(min=1.0)

    mse = numer / denom
    return mse
