def get_number_of_nodes(batch_tensor,
                        batch_dim,
                        index):
    """Returns the number of nodes per sample for a batch tensor."""
    num_nodes = (batch_tensor == index).nonzero(as_tuple=True)[0]

    if len(num_nodes) == 0:
        num_nodes = batch_tensor.shape[batch_dim]
    else:
        num_nodes = num_nodes[0].item()
    return num_nodes