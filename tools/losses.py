import torch


def nt_xent_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Compute NT-Xent (contrastive) loss for a batch with potentially multiple positives per anchor.

    embeddings: Tensor[B, D]
    labels: Tensor[B] (int labels)
    Returns: scalar loss
    """
    if embeddings.dim() != 2:
        embeddings = embeddings.view(embeddings.size(0), -1)

    device = embeddings.device
    labels = labels.to(device)

    # normalize embeddings
    z = torch.nn.functional.normalize(embeddings, dim=1)

    # similarity matrix
    sim = torch.matmul(z, z.t()) / temperature 

    # mask out self-similarity
    diag_mask = torch.eye(sim.size(0), device=device).bool()

    # compute logsumexp over all except self
    sim_masked = sim.masked_fill(diag_mask, float('-inf'))
    logsumexp_all = torch.logsumexp(sim_masked, dim=1) 

    # compute logsumexp over positives only
    loss_terms = []
    for i in range(labels.size(0)):
        pos_mask = (labels == labels[i]).to(device)
        pos_mask[i] = False
        if pos_mask.sum() == 0:
            continue
        sim_pos = sim[i][pos_mask]
        logsumexp_pos = torch.logsumexp(sim_pos, dim=0)
        loss_i = -(logsumexp_pos - logsumexp_all[i])
        loss_terms.append(loss_i)

    if len(loss_terms) == 0:
        return torch.tensor(0.0, device=device)

    loss = torch.stack(loss_terms).mean()
    return loss
