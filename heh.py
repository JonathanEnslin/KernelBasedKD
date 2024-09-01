import torch
import numpy as np

def calculate_entropy(values, num_bins=100):
    # Flatten the values
    flattened = values.view(-1).detach().cpu().numpy()
    
    # Create a histogram (empirical PDF) of the values
    hist, bin_edges = np.histogram(flattened, bins=num_bins, density=True)
    
    # Normalize the histogram to get probabilities
    p = hist / np.sum(hist)
    
    # Calculate entropy
    entropy = -np.sum(p * np.log(p + 1e-9))  # Adding a small value to avoid log(0)
    
    return entropy

def compute_filter_metrics(filter_weights, num_bins=100):
    # Flatten the filter to compute vector-based metrics
    flattened = filter_weights.view(-1)
    
    # Compute basic statistics
    mean = filter_weights.mean().item()
    abs_mean = filter_weights.abs().mean().item()

    mean_dim0 = filter_weights.mean(dim=0)
    mean_dim1 = filter_weights.mean(dim=1)
    mean_dim01 = filter_weights.mean(dim=(0,1))
    
    abs_mean_dim0 = filter_weights.abs().mean(dim=0)
    abs_mean_dim1 = filter_weights.abs().mean(dim=1)
    abs_mean_dim01 = filter_weights.abs().mean(dim=(0,1))
    
    # Compute angles
    if flattened.is_complex():
        angles = torch.atan2(flattened.imag, flattened.real)
    else:
        angles = torch.atan2(flattened, torch.ones_like(flattened))
    
    angles_mean = angles.mean().item()
    angles_std = angles.std().item()
    
    # Compute norms
    l2_norm = flattened.norm(2).item()  # Euclidean norm
    max_norm = flattened.norm(float('inf')).item()  # Max norm
    
    # Compute sparsity (percentage of zeros)
    sparsity = (flattened == 0).float().mean().item()

    # Compute overall entropy
    entropy = calculate_entropy(filter_weights, num_bins=num_bins)
    
    # Compute entropy map for each (h, w) position across C_in and C_out
    H, W = filter_weights.shape[2], filter_weights.shape[3]
    entropy_map = np.zeros((H, W))
    
    for h in range(H):
        for w in range(W):
            position_values = filter_weights[:, :, h, w]
            entropy_map[h, w] = calculate_entropy(position_values, num_bins=num_bins)
    
    # Parameterized dict
    metrics = {
        'mean': mean,
        'abs_mean': abs_mean,
        'mean_dim0': mean_dim0.numpy().tolist(),
        'mean_dim1': mean_dim1.numpy().tolist(),
        'mean_dim01': mean_dim01.numpy().tolist(),
        'abs_mean_dim0': abs_mean_dim0.numpy().tolist(),
        'abs_mean_dim1': abs_mean_dim1.numpy().tolist(),
        'abs_mean_dim01': abs_mean_dim01.numpy().tolist(),
        'angles_mean': angles_mean,
        'angles_std': angles_std,
        'l2_norm': l2_norm,
        'max_norm': max_norm,
        'sparsity': sparsity,
        'entropy': entropy,
        'entropy_map': entropy_map.tolist()  # Entropy at each (h, w) position
    }
    
    return metrics

def collect_metrics_for_network_layers(layers_weights):
    metrics_list = []
    for layer_index, layer_weights in enumerate(layers_weights):
        if len(layer_weights.shape) == 4:  # Assuming weights have the shape [C_out, C_in, H, W]
            metrics = compute_filter_metrics(layer_weights)
            metrics_list.append({
                'layer_index': layer_index,
                'metrics': metrics
            })
        else:
            print(f"Skipping layer {layer_index} as it does not have a 4D shape.")
    
    return metrics_list

# Example usage with a list of layer weights:
# Assuming `model` is your network and you extract weights as a list
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 6, 3),
    torch.nn.Conv2d(6, 16, 3)
)

layers_weights = [param.data for param in model.parameters() if param.requires_grad and len(param.data.shape) == 4]
metrics_list = collect_metrics_for_network_layers(layers_weights)
print(metrics_list)
