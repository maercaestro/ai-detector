import torch
import torch.nn as nn
import numpy as np

class SRMLayer(nn.Module):
    """
    A fixed convolutional layer initialized with 30 SRM filters.
    This serves as the 'preprocessing' step for the SSP ResNet.
    Ref: Chen et al., "Single Simple Patch is All You Need".
    """
    def __init__(self):
        super(SRMLayer, self).__init__()
        self.out_channels = 30
        self.kernel_size = 5
        # Define the Conv2d layer - No bias, fixed weights
        self.conv = nn.Conv2d(3, self.out_channels, self.kernel_size, 
                              stride=1, padding=2, bias=False)
        
        # Load SRM filters (Placeholder for the full 30 filters)
        # In a production environment, you would load the full 5x5 kernels
        # derived from the standard steganalysis toolset.
        # Below we demonstrate constructing 3 basic types for illustration.
        srm_weights = self._get_srm_weights()
        
        # Set weights and freeze them (requires_grad=False)
        self.conv.weight.data = torch.from_numpy(srm_weights).float()
        for param in self.conv.parameters():
            param.requires_grad = False

    def _get_srm_weights(self):
        # Initialize (30 filters, 3 input channels, 5, 5)
        weights = np.zeros((30, 3, 5, 5))
        
        # Example 1: KV Filter (Residual)
        kv = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                       [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]) / 12.0
        
        # Example 2: Edge 3x3 (Padded to 5x5)
        edge = np.array([, , [0,1,-4,1,0], 
                         , ])
        
        # Populate weights (replicating to 30 for shape consistency in this demo)
        # In reality, you load 30 DISTINCT filters.
        for i in range(30):
            # Apply to all 3 RGB channels similarly or independently depending on paper variant
            # SSP typically treats RGB channels equally or converts to Y approach.
            weights[i, 0, :, :] = kv if i % 2 == 0 else edge
            weights[i, 1, :, :] = kv if i % 2 == 0 else edge
            weights[i, 2, :, :] = kv if i % 2 == 0 else edge
            
        return weights

    def forward(self, x):
        return self.conv(x)