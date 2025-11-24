def select_simplest_patch(image_tensor: torch.Tensor, patch_size: int = 64, num_candidates: int = 50) -> torch.Tensor:
    """
    Selects the patch with the lowest pixel variance.
    Input: Image Tensor (1, 3, H, W) normalized 
    Output: Selected Patch (1, 3, patch_size, patch_size)
    """
    b, c, h, w = image_tensor.shape
    
    min_var = float('inf')
    best_patch = None
    
    # Random Sampling Strategy (Stochastic Search)
    # This is often faster and sufficiently accurate compared to sliding window
    for _ in range(num_candidates):
        y = np.random.randint(0, h - patch_size)
        x = np.random.randint(0, w - patch_size)
        
        patch = image_tensor[:, :, y:y+patch_size, x:x+patch_size]
        
        # Calculate Variance (Texture Diversity)
        # We calculate mean variance across channels
        variance = torch.var(patch).item()
        
        if variance < min_var:
            min_var = variance
            best_patch = patch
            
    # Resize to 256x256 as required by ResNet-50 input in SSP paper
    # Ref: Snippet [20] "Bigger it (256x256)"
    if best_patch is not None:
        best_patch_resized = F.interpolate(best_patch, size=(256, 256), 
                                          mode='bilinear', align_corners=False)
        return best_patch_resized
    else:
        raise ValueError("Could not extract patch")