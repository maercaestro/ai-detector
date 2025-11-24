import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List

class PatchCraftPreprocessor:
    """
    Implements the Smash & Reconstruction algorithm for PatchCraft.
    Ref: Zhong et al., "Rich and Poor Texture Contrast" (2024).
    """
    def __init__(self, patch_size: int = 32, partition_ratio: float = 0.5):
        self.patch_size = patch_size
        self.partition_ratio = partition_ratio

    def calculate_richness_metric(self, image: np.ndarray) -> np.ndarray:
        """
        Calculates the texture richness using Gradient Magnitude (Sobel).
        Input: Grayscale Image (H, W)
        Output: Gradient Magnitude Map (H, W)
        """
        # Compute gradients along X and Y axis
        # Using 64-bit float to prevent overflow during calculation
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude: sqrt(Gx^2 + Gy^2)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return magnitude

    def smash_and_reconstruct(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits image into Rich and Poor composite images.
        """
        # 1. Load Image and Convert to Grayscale for Metric Calculation
        # Color is preserved for the final patches, but metric is luma-based.
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not load image at {image_path}")
        
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w, c = img_bgr.shape

        # 2. Grid Handling - Trim image to be divisible by patch_size
        h_trim = h - (h % self.patch_size)
        w_trim = w - (w % self.patch_size)
        img_bgr = img_bgr[:h_trim, :w_trim, :]
        img_gray = img_gray[:h_trim, :w_trim]

        # 3. Extract Patches and Calculate Metrics
        patches =
        metrics =
        
        grad_map = self.calculate_richness_metric(img_gray)

        for y in range(0, h_trim, self.patch_size):
            for x in range(0, w_trim, self.patch_size):
                # Extract patch
                patch_img = img_bgr[y:y+self.patch_size, x:x+self.patch_size, :]
                patch_metric_val = np.mean(grad_map[y:y+self.patch_size, x:x+self.patch_size])
                
                patches.append(patch_img)
                metrics.append(patch_metric_val)

        # 4. Sort Patches based on Texture Richness
        # Zip, Sort, and Unzip
        sorted_pairs = sorted(zip(metrics, patches), key=lambda pair: pair)
        sorted_patches = [p for m, p in sorted_pairs]
        
        # 5. Partition into Rich and Poor Sets
        total_patches = len(sorted_patches)
        split_idx = int(total_patches * self.partition_ratio)
        
        # Poor patches have LOW gradient magnitude (start of list)
        # Rich patches have HIGH gradient magnitude (end of list)
        poor_set = sorted_patches[:split_idx]
        rich_set = sorted_patches[total_patches - split_idx:] # Take top N
        
        # 6. Reconstruct Composite Images
        # We verify that we have patches to reconstruction
        if not poor_set or not rich_set:
            raise ValueError("Partitioning resulted in empty sets. Check patch size/image size.")

        img_poor = self._reassemble(poor_set)
        img_rich = self._reassemble(rich_set)

        return img_rich, img_poor

    def _reassemble(self, patch_list: List[np.ndarray]) -> np.ndarray:
        """
        Helper to stitch a list of patches into a square-ish image.
        """
        count = len(patch_list)
        # Calculate grid dimensions (approximate square)
        grid_width = int(np.ceil(np.sqrt(count)))
        grid_height = int(np.ceil(count / grid_width))
        
        # Create canvas
        full_h = grid_height * self.patch_size
        full_w = grid_width * self.patch_size
        canvas = np.zeros((full_h, full_w, 3), dtype=np.uint8)
        
        for idx, patch in enumerate(patch_list):
            row = idx // grid_width
            col = idx % grid_width
            
            y = row * self.patch_size
            x = col * self.patch_size
            
            canvas[y:y+self.patch_size, x:x+self.patch_size, :] = patch
            
        return canvas