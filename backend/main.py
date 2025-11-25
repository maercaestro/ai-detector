from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import numpy as np
from scipy import fftpack
from scipy.stats import gennorm
from typing import Dict, List, Any, Optional
import json
from openai import OpenAI
import os
import shutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import database

# Load environment variables from .env file
load_dotenv()

# Create uploads directory if it doesn't exist
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="AI Image Detector API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    database.init_database()
    print("🚀 Server started successfully")



def get_azimuthal_average(image_bytes: bytes) -> tuple[List[float], List[float]]:
    """
    Analyzes the azimuthal power spectrum of an image.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Tuple of (normalized_frequency, radial_profile) lists
    """
    # Decode image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Invalid image format")
    
    # FFT and Radial Profile Calculation
    f = fftpack.fft2(img)
    fshift = fftpack.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    r = np.sqrt(x**2 + y**2).astype(int)
    
    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / (nr + 1e-8)
    
    # Normalize for plotting
    normalized_freq = np.linspace(0, 1, len(radial_profile))
    return normalized_freq.tolist(), radial_profile.tolist()


def get_ggd_parameters(data: np.ndarray) -> float:
    """
    Fits Generalized Gaussian Distribution and returns the Shape (Beta) parameter.
    
    Args:
        data: Input data array
        
    Returns:
        Shape parameter (beta)
    """
    if len(data) < 100:
        return 0.0
    try:
        shape, loc, scale = gennorm.fit(data)
        return float(shape)
    except:
        return 0.0


def analyze_patchcraft(img: np.ndarray, patch_size: int = 32, partition_ratio: float = 0.5) -> Dict:
    """
    PatchCraft: Analyzes texture consistency using rich/poor patch comparison.
    Splits image by gradient magnitude and compares SRM filter responses.
    
    Args:
        img: Grayscale or color image
        patch_size: Size of patches to extract
        partition_ratio: Ratio for rich/poor split (0.5 = top/bottom 50%)
        
    Returns:
        Dictionary with patchcraft_score and component features
    """
    try:
        # Handle color images
        if img.ndim == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_color = img
        else:
            img_gray = img
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        h, w = img_gray.shape
        
        # Trim to be divisible by patch_size
        h_trim = h - (h % patch_size)
        w_trim = w - (w % patch_size)
        img_gray = img_gray[:h_trim, :w_trim]
        img_color = img_color[:h_trim, :w_trim]
        
        # Calculate gradient magnitude (texture richness metric)
        grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Extract patches and their metrics
        patches = []
        metrics = []
        
        for y in range(0, h_trim, patch_size):
            for x in range(0, w_trim, patch_size):
                patch = img_color[y:y+patch_size, x:x+patch_size]
                metric_val = np.mean(grad_magnitude[y:y+patch_size, x:x+patch_size])
                patches.append(patch)
                metrics.append(metric_val)
        
        if len(patches) < 4:
            return {"patchcraft_score": 0.0, "rich_feature": 0.0, "poor_feature": 0.0}
        
        # Sort patches by gradient magnitude
        sorted_pairs = sorted(zip(metrics, patches), key=lambda p: p[0])
        sorted_patches = [p for m, p in sorted_pairs]
        
        # Partition into rich (high gradient) and poor (low gradient)
        total = len(sorted_patches)
        split_idx = max(1, int(total * partition_ratio))
        
        poor_patches = sorted_patches[:split_idx]
        rich_patches = sorted_patches[total - split_idx:]
        
        # Define SRM KV filter
        kv_kernel = np.array([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]
        ], dtype=np.float32) / 12.0
        
        # Apply SRM filter and extract features
        def extract_srm_feature(patch_list):
            features = []
            for patch in patch_list:
                if patch.ndim == 3:
                    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                else:
                    patch_gray = patch
                residual = cv2.filter2D(patch_gray.astype(np.float32), -1, kv_kernel)
                features.append(np.std(residual))
            return np.mean(features) if features else 0.0
        
        rich_feature = extract_srm_feature(rich_patches)
        poor_feature = extract_srm_feature(poor_patches)
        
        # Discrepancy score (higher = more inconsistent = more AI-like)
        patchcraft_score = abs(rich_feature - poor_feature)
        
        return {
            "patchcraft_score": float(patchcraft_score),
            "rich_feature": float(rich_feature),
            "poor_feature": float(poor_feature)
        }
    except Exception as e:
        print(f"PatchCraft error: {e}")
        return {"patchcraft_score": 0.0, "rich_feature": 0.0, "poor_feature": 0.0}


def analyze_spectral_grid(img: np.ndarray) -> Dict:
    """
    Detects Fourier grid artifacts at specific periods (4, 8, 16 pixels).
    Common in Flux.1 and other VAE-based generators.
    
    Args:
        img: Grayscale or color image
        
    Returns:
        Dictionary with spectral_score and detected periods
    """
    try:
        # Convert to grayscale if needed
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Compute 2D FFT
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)
        
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # Check for peaks at specific harmonic periods
        period_scores = {}
        
        for factor in [4, 8, 16]:
            step_x = max(1, w // factor)
            step_y = max(1, h // factor)
            
            factor_energy = 0
            count = 0
            
            # Sample grid points for this factor
            for i in range(1, min(factor, 10)):
                for j in range(1, min(factor, 10)):
                    py = center_y + (i - factor // 2) * step_y
                    px = center_x + (j - factor // 2) * step_x
                    
                    if 0 <= py < h and 0 <= px < w:
                        # Compare peak to local neighborhood
                        window_size = 5
                        y1, y2 = max(0, py - window_size), min(h, py + window_size)
                        x1, x2 = max(0, px - window_size), min(w, px + window_size)
                        
                        local_region = magnitude_spectrum[y1:y2, x1:x2]
                        peak_val = magnitude_spectrum[py, px]
                        avg_val = np.mean(local_region)
                        
                        # Detect significant peaks
                        if peak_val > avg_val * 1.1:
                            factor_energy += (peak_val - avg_val)
                            count += 1
            
            period_scores[f"period_{factor}"] = float(factor_energy / count) if count > 0 else 0.0
        
        # Overall spectral score (max across periods)
        spectral_score = max(period_scores.values()) if period_scores else 0.0
        
        return {
            "spectral_score": float(spectral_score),
            **period_scores
        }
    except Exception as e:
        print(f"Spectral analysis error: {e}")
        return {"spectral_score": 0.0, "period_4": 0.0, "period_8": 0.0, "period_16": 0.0}


def analyze_chromatic_aberration(img_color: np.ndarray) -> Dict:
    """
    Detects chromatic aberration (color fringing) at image edges.
    Real optical lenses show spatial misalignment between color channels.
    AI images typically have perfectly aligned channels.
    
    Uses radial distance method: measures how R-B channel difference 
    increases toward image corners.
    
    Args:
        img_color: BGR color image
        
    Returns:
        Dictionary with displacement metrics
    """
    try:
        if img_color is None or img_color.size == 0:
            return {"lca_displacement": 0.0, "lca_found": False}
        
        # Split color channels
        b, g, r = cv2.split(img_color)
        
        h, w = img_color.shape[:2]
        
        # Convert to float for precision
        r_float = r.astype(np.float32)
        b_float = b.astype(np.float32)
        
        # Calculate channel difference (chromatic aberration signal)
        rb_diff = np.abs(r_float - b_float)
        
        # Create radial distance map from center
        center_y, center_x = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        radial_distance = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        max_radius = np.sqrt(center_y**2 + center_x**2)
        radial_distance_norm = radial_distance / max_radius  # 0 at center, 1 at corners
        
        # Divide image into radial zones
        zone1_mask = radial_distance_norm < 0.3  # Center region
        zone2_mask = (radial_distance_norm >= 0.3) & (radial_distance_norm < 0.7)  # Mid region
        zone3_mask = radial_distance_norm >= 0.7  # Edge/corner region
        
        # Calculate mean chromatic difference in each zone
        # Only consider pixels with enough edge content (gradient > threshold)
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Only analyze high-gradient areas (edges) where CA is visible
        edge_threshold = np.percentile(gradient_mag, 60)
        edge_mask = gradient_mag > edge_threshold
        
        def calculate_zone_ca(zone_mask):
            combined_mask = zone_mask & edge_mask
            if np.sum(combined_mask) < 100:  # Need enough pixels
                return 0.0
            return float(np.mean(rb_diff[combined_mask]))
        
        ca_center = calculate_zone_ca(zone1_mask)
        ca_mid = calculate_zone_ca(zone2_mask)
        ca_edge = calculate_zone_ca(zone3_mask)
        
        print(f"CA analysis - Center: {ca_center:.2f}, Mid: {ca_mid:.2f}, Edge: {ca_edge:.2f}")
        
        # Real lenses show increasing CA toward edges
        # Calculate radial gradient of chromatic aberration
        ca_increase = (ca_edge - ca_center)
        
        # Normalize by image intensity to get relative measure
        mean_intensity = float(np.mean(gray))
        if mean_intensity > 0:
            ca_increase_normalized = (ca_increase / mean_intensity) * 100  # As percentage
        else:
            ca_increase_normalized = 0.0
        
        print(f"CA radial increase: {ca_increase_normalized:.3f}%")
        
        # Real photos: CA increases toward edges (positive value, typically 0.5-2%)
        # AI images: CA flat (near 0)
        # Smartphones: CA near 0 or slightly negative (ISP correction)
        
        # Use absolute value for scoring - both negative (overcorrection) and positive (natural) are real
        ca_absolute = abs(ca_increase_normalized)
        
        return {
            "lca_displacement": float(ca_increase_normalized),  # Keep original sign for reasoning
            "lca_absolute": float(ca_absolute),  # Absolute value for scoring
            "lca_found": True,
            "ca_center": float(ca_center),
            "ca_edge": float(ca_edge)
        }
        
    except Exception as e:
        print(f"Chromatic aberration analysis error: {e}")
        import traceback
        traceback.print_exc()
        return {"lca_displacement": 0.0, "lca_found": False}


def analyze_local_consistency(img: np.ndarray, block_size: int = 64) -> Dict:
    """
    Analyzes local patches to detect inconsistent noise patterns.
    
    AI images often have patches with Gaussian-like noise (β ≈ 2.0) in synthetic textures,
    while real photos maintain consistent natural noise (β < 1.5) across regions.
    
    Args:
        img: Grayscale image
        block_size: Size of each patch (default 64x64)
        
    Returns:
        Dictionary with heatmap data and statistics
    """
    h, w = img.shape
    
    # Calculate number of blocks
    n_blocks_h = h // block_size
    n_blocks_w = w // block_size
    
    if n_blocks_h == 0 or n_blocks_w == 0:
        return {
            "heatmap": [],
            "suspicious_patches": 0,
            "variance": 0,
            "mean_beta": 0
        }
    
    # Store beta values for each patch
    beta_map = np.zeros((n_blocks_h, n_blocks_w))
    
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            # Extract patch
            y_start = i * block_size
            y_end = y_start + block_size
            x_start = j * block_size
            x_end = x_start + block_size
            
            patch = img[y_start:y_end, x_start:x_end]
            
            # Denoise patch to get residual
            try:
                patch_clean = cv2.fastNlMeansDenoising(patch, None, h=10, templateWindowSize=7, searchWindowSize=21)
                residual = patch.astype(float) - patch_clean.astype(float)
                residual_data = residual.flatten()
                residual_data = residual_data[residual_data != 0]
                
                # Get beta for this patch
                beta = get_ggd_parameters(residual_data)
                beta_map[i, j] = beta if beta > 0 else 1.0  # Default to 1.0 if failed
            except:
                beta_map[i, j] = 1.0
    
    # Calculate statistics
    valid_betas = beta_map[beta_map > 0]
    
    if len(valid_betas) == 0:
        return {
            "heatmap": beta_map.tolist(),
            "suspicious_patches": 0,
            "variance": 0,
            "mean_beta": 0,
            "max_beta": 0,
            "min_beta": 0
        }
    
    # Count suspicious patches (AI-like, β > 1.65)
    # Uncertain range: 1.0-1.65, AI: > 1.65
    suspicious_patches = np.sum(valid_betas > 1.65)
    suspicious_ratio = suspicious_patches / len(valid_betas)
    
    return {
        "heatmap": beta_map.tolist(),
        "suspicious_patches": int(suspicious_patches),
        "suspicious_ratio": float(suspicious_ratio),
        "variance": float(np.var(valid_betas)),
        "mean_beta": float(np.mean(valid_betas)),
        "max_beta": float(np.max(valid_betas)),
        "min_beta": float(np.min(valid_betas)),
        "blocks_analyzed": int(len(valid_betas))
    }


def analyze_texture_consistency(img, patch_size=32):
    """
    Analyzes texture consistency across rich vs poor texture regions.
    AI images show inconsistent noise patterns between high/low texture areas.
    
    Real photos: Inconsistency < 0.2 (consistent noise across textures)
    AI images: Inconsistency > 0.5 (inconsistent synthesis patterns)
    
    Args:
        img: Grayscale image
        patch_size: Size of patches to analyze
        
    Returns:
        Dictionary with rich_beta, poor_beta, and inconsistency score
    """
    # Extract Noise Residual
    img_clean = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    residual = img.astype(float) - img_clean.astype(float)
    
    h, w = residual.shape
    patches = []
    variances = []

    # Extract Patches & Calculate Variance (Texture Richness)
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = residual[y:y+patch_size, x:x+patch_size].flatten()
            patch = patch[patch != 0]
            
            if len(patch) > 100:
                var = np.var(patch)
                patches.append(patch)
                variances.append(var)

    if not patches:
        return {
            "rich_beta": 1.0,
            "poor_beta": 1.0,
            "inconsistency": 0.0
        }

    # Sort into Rich vs Poor (Top 30% vs Bottom 30%)
    sorted_indices = np.argsort(variances)
    n = len(patches)
    n_select = max(1, int(n * 0.3))
    
    poor_indices = sorted_indices[:n_select]
    rich_indices = sorted_indices[-n_select:]
    
    # Combine data from all patches in each group
    poor_data = np.concatenate([patches[i] for i in poor_indices])
    rich_data = np.concatenate([patches[i] for i in rich_indices])

    # Fit GGD to both groups
    beta_poor = get_ggd_parameters(poor_data)
    beta_rich = get_ggd_parameters(rich_data)
    
    # The "Inconsistency Score"
    inconsistency = abs(beta_rich - beta_poor)
    
    return {
        "rich_beta": float(beta_rich),
        "poor_beta": float(beta_poor),
        "inconsistency": float(inconsistency)
    }


def analyze_simplest_patch(img, patch_size=32):
    """
    Analyzes the simplest (lowest variance) patch to detect AI synthesis in smooth areas.
    This is the "Simplest Smooth Patch" (SSP) method.
    
    Real photos: β ≈ 0.6-0.9 (sensor noise in sky/walls)
    AI images: β ≈ 2.0 (Gaussian remnants) or > 5.0 (over-smoothed/quantized)
    
    Args:
        img: Grayscale image
        patch_size: Size of patches to analyze
        
    Returns:
        Dictionary with ssp_beta and patch variance
    """
    # Find the "Simplest" Patch (Lowest Variance)
    min_var = float('inf')
    best_patch = None
    best_coords = None
    
    h, w = img.shape
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = img[y:y+patch_size, x:x+patch_size]
            var = np.var(patch)
            
            # We want the patch with the LOWEST variance (flattest area)
            # Lowered threshold to 0.1 to allow smooth areas like sky
            if var < min_var and var > 0.1:
                min_var = var
                best_patch = patch
                best_coords = (y, x)

    if best_patch is None:
        print("SSP: No valid patch found (all patches too uniform)")
        return {
            "ssp_beta": 1.0,
            "ssp_variance": 0.0,
            "found_patch": False
        }

    # Extract Noise from that Patch using Gaussian blur
    patch_clean = cv2.GaussianBlur(best_patch, (3, 3), 0)
    noise = best_patch.astype(float) - patch_clean.astype(float)
    noise = noise.flatten()
    
    # Keep all noise values including zeros - they're valid for smooth patches
    # Just need enough samples for GGD fitting
    if len(noise) < 20:
        print(f"SSP: Not enough noise samples ({len(noise)} < 20)")
        return {
            "ssp_beta": 1.0,
            "ssp_variance": float(min_var),
            "found_patch": False
        }
    
    if best_coords:
        print(f"SSP: Found patch at ({best_coords[0]},{best_coords[1]}), variance={min_var:.3f}, noise_samples={len(noise)}")

    # Fit GGD to this specific noise
    beta_ssp = get_ggd_parameters(noise)
    
    return {
        "ssp_beta": float(beta_ssp),
        "ssp_variance": float(min_var),
        "found_patch": True
    }


def get_ai_reasoning(metrics_data: Dict) -> Dict:
    """
    Uses GPT-4o-mini model to verify detection results.
    Provides qualitative analysis based on physics and statistics.
    
    Args:
        metrics_data: Dictionary containing all detection metrics
        
    Returns:
        Dictionary with AI reasoning verdict
    """
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        prompt = f"""You are the "Forensic Adjudicator," a specialized AI designed to detect synthetic imagery. You do not guess; you analyze statistical and physical anomalies based on a 7-Branch Weighted Fusion System.

Your goal is to distinguish between three categories:
1. **AUTHENTIC (Real)**: Raw or minimally processed sensor data.
2. **COMPUTATIONAL (Smartphone)**: Real optical data heavily processed by ISPs (Samsung/Apple algorithms).
3. **SYNTHETIC (AI)**: Generated by Diffusion models (Midjourney, Flux, Stable Diffusion).

### THE HIERARCHY OF EVIDENCE (Logic Flow)
You must evaluate the provided metrics in this specific order. "Physics" overrides "Statistics."

**STEP 1: The Physics Check (Branch 7 - CRITICAL)**
- **Chromatic Aberration (CA):** Real lenses MUST have radial CA (> 0.1%).
- **IF CA < 0.1%:** The image is likely FAKE, regardless of noise levels.
- **IF CA > 3.0%:** Suspicious, but could be a cheap wide-angle lens.
- **IF CA is 0.5% - 2.0%:** Strong evidence of a physical lens.

**STEP 2: The "Smartphone Trap" (Branch 1, 2, & SSP)**
- Smartphones often have High Beta (β > 2.0) due to denoising. Do not confuse this with AI.
- **Differentiation Rule:**
  - High Beta (Smooth) + Valid CA (Lens) = **SMARTPHONE (Real)**.
  - High Beta (Smooth) + No CA (No Lens) = **AI (Fake)**.

**STEP 3: The Artifact Check (Branch 4, 5)**
- **Spectral Grid (Branch 5):**
  - Score > 25 indicates a grid.
  - If CA is Valid, this grid is likely JPEG/HEIC compression.
  - If CA is Invalid, this grid is a VAE/Diffusion artifact.

**STEP 4: Texture Logic (Branch 3)**
- **Consistency:** AI often hallucinates detailed textures (Rich) but smoothes backgrounds (Poor).
- High Inconsistency (> 0.5) with Gaussian Noise supports an AI verdict.

### INPUT DATA
{json.dumps(metrics_data, indent=2)}

### OUTPUT FORMAT (JSON)
Return ONLY valid JSON with this structure:
{{
  "verdict": "REAL" | "FAKE" | "UNCERTAIN",
  "sub_category": "DSLR_Raw" | "Smartphone_Computational" | "Generative_AI" | "Digital_Art",
  "confidence_score": 0-100,
  "primary_smoking_gun": "The specific metric that sealed the verdict.",
  "reasoning_chain": [
    "Step 1: Analyzed Optics...",
    "Step 2: Analyzed Noise Statistics...",
    "Step 3: Evaluated Consistency...",
    "Conclusion..."
  ],
  "human_readable_explanation": "A 2-sentence summary for a non-technical user."
}}"""

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.7,
            max_tokens=2048,
            top_p=1,
            stream=False,
            stop=None
        )
        
        response_text = completion.choices[0].message.content
        
        # Parse JSON response
        # Remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        reasoning_result = json.loads(response_text)
        return reasoning_result
        
    except Exception as e:
        print(f"AI Reasoning error: {e}")
        return {
            "verdict": "UNCERTAIN",
            "sub_category": "Error",
            "confidence_score": 0,
            "primary_smoking_gun": "Reasoning service unavailable",
            "reasoning_chain": ["Error occurred during AI reasoning"],
            "human_readable_explanation": "Unable to perform AI reasoning verification."
        }


def detect_ai_generation(frequency: List[float], power: List[float], image_bytes: bytes) -> Dict:
    """
    Detects AI-generated images using Generalized Gaussian Distribution (GGD) analysis.
    
    This method analyzes noise characteristics:
    - Real photos: GGD shape (beta) typically 0.5-1.5 (heavy tails, sharp peak)
    - AI-generated: GGD shape closer to 2.0 (Gaussian-like, synthetic noise)
    
    Args:
        frequency: Normalized frequency values (for display)
        power: Power spectrum values (for display)
        image_bytes: Raw image bytes for noise analysis
        
    Returns:
        Detection result with verdict and confidence
    """
    try:
        # Decode image in COLOR for advanced analysis
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        if img is None or img_color is None:
            raise ValueError("Invalid image format")
        
        # === FEATURE 1: NOISE RESIDUAL GGD ===
        # Extract noise residual using Non-Local Means Denoising
        # This separates the "clean" signal from noise
        img_clean = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
        residual = img.astype(float) - img_clean.astype(float)
        
        # Flatten residual and remove zeros (border artifacts)
        data = residual.flatten()
        data = data[data != 0]
        
        if len(data) < 100:
            raise ValueError("Insufficient noise data for analysis")
        
        # Fit Generalized Gaussian Distribution
        # Returns: (shape/beta, location, scale)
        shape, loc, scale = gennorm.fit(data)
        
        # Calculate additional statistics
        kurtosis = np.mean((data - np.mean(data))**4) / (np.var(data)**2)
        noise_std = np.std(data)
        
        # Prepare histogram data for visualization
        # Clip to reasonable range for better visualization
        data_clipped = np.clip(data, -50, 50)
        hist, bin_edges = np.histogram(data_clipped, bins=100, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Generate GGD fit curve
        x_range = np.linspace(-50, 50, 200)
        ggd_curve = gennorm.pdf(x_range, shape, loc, scale)
        
        # Also generate ideal Gaussian for comparison
        gaussian_curve = gennorm.pdf(x_range, 2.0, loc, scale)  # beta=2.0 is pure Gaussian
        
        # === FEATURE 2: LOCAL CONSISTENCY HEATMAP ===
        # Analyze patches to detect localized AI artifacts
        print("Starting local consistency analysis...")
        local_analysis = analyze_local_consistency(img, block_size=64)
        print(f"Local analysis result: {type(local_analysis)}")
        if local_analysis:
            print(f"Heatmap shape: {len(local_analysis.get('heatmap', []))}x{len(local_analysis.get('heatmap', [[]])[0]) if local_analysis.get('heatmap') else 0}")
            print(f"Mean beta: {local_analysis.get('mean_beta', 'N/A')}")
        
        # === FEATURE 3: TEXTURE CONSISTENCY ANALYSIS ===
        # Compares noise in rich vs poor texture regions
        print("Starting texture consistency analysis...")
        texture_analysis = analyze_texture_consistency(img, patch_size=32)
        print(f"Texture analysis - Rich β: {texture_analysis['rich_beta']:.3f}, Poor β: {texture_analysis['poor_beta']:.3f}, Inconsistency: {texture_analysis['inconsistency']:.3f}")
        
        # === FEATURE 4: SIMPLEST SMOOTH PATCH (SSP) ANALYSIS ===
        # Analyzes the flattest region (sky, walls, etc.) for synthesis artifacts
        print("Starting simplest patch analysis...")
        ssp_analysis = analyze_simplest_patch(img, patch_size=32)
        print(f"SSP analysis - β: {ssp_analysis['ssp_beta']:.3f}, Variance: {ssp_analysis['ssp_variance']:.2f}, Found: {ssp_analysis['found_patch']}")
        
        # === FEATURE 5: PATCHCRAFT ANALYSIS ===
        # Compares SRM filter responses between rich and poor texture patches
        print("Starting PatchCraft analysis...")
        patchcraft_analysis = analyze_patchcraft(img_color, patch_size=32)
        print(f"PatchCraft - Score: {patchcraft_analysis['patchcraft_score']:.3f}, Rich: {patchcraft_analysis['rich_feature']:.3f}, Poor: {patchcraft_analysis['poor_feature']:.3f}")
        
        # === FEATURE 6: SPECTRAL GRID ANALYSIS ===
        # Detects Fourier grid artifacts at specific periods
        print("Starting Spectral Grid analysis...")
        spectral_analysis = analyze_spectral_grid(img)
        print(f"Spectral Grid - Score: {spectral_analysis['spectral_score']:.2f}, P4: {spectral_analysis['period_4']:.2f}, P8: {spectral_analysis['period_8']:.2f}, P16: {spectral_analysis['period_16']:.2f}")
        
        # === FEATURE 7: CHROMATIC ABERRATION ===
        # Real lenses show color fringing at edges (channel misalignment)
        print("Starting Chromatic Aberration analysis...")
        lca_analysis = analyze_chromatic_aberration(img_color)
        print(f"Chromatic Aberration - Displacement: {lca_analysis['lca_displacement']:.4f} pixels, Found: {lca_analysis['lca_found']}")
        
        # Analyze frequency spectrum characteristics
        freq_array = np.array(frequency)
        power_array = np.array(power)
        
        high_freq_mask = freq_array > 0.7
        high_freq_power = power_array[high_freq_mask]
        mid_freq_power = power_array[(freq_array > 0.3) & (freq_array <= 0.7)]
        
        high_freq_slope = 0
        if len(high_freq_power) > 1:
            high_freq_slope = np.polyfit(freq_array[high_freq_mask], high_freq_power, 1)[0]
        
        power_ratio = np.mean(mid_freq_power) / (np.mean(high_freq_power) + 1e-8) if len(high_freq_power) > 0 else 1.0
        
        # === HYBRID SCORING SYSTEM ===
        # Combine multiple detection methods with weighted fusion
        branch_scores = {}
        reasons = []
        
        # Branch 1: GGD Global Noise (Weight: 25%)
        ggd_score = 0
        if 1.8 <= shape <= 2.2:
            ggd_score = 100
            reasons.append(f"Gaussian-like noise (β={shape:.2f})")
        elif shape > 1.5:
            ggd_score = 62
        elif shape > 1.3:
            ggd_score = 30
        
        # Noise level analysis - both too low and too high are suspicious
        # Real camera photos: noise_std typically 3.0-15.0
        # AI images: often <1.0 (over-smoothed) or >20.0 (artificial noise injection)
        if noise_std < 1.0:
            ggd_score = min(ggd_score + 40, 100)
            reasons.append(f"Abnormally low noise (std={noise_std:.3f} - over-smoothed)")
        elif noise_std < 2.0:
            ggd_score = min(ggd_score + 20, 100)
            reasons.append(f"Very low noise (std={noise_std:.3f})")
        elif noise_std > 25.0:
            ggd_score = min(ggd_score + 30, 100)
            reasons.append(f"Abnormally high noise (std={noise_std:.3f} - artificial)")
        elif noise_std > 20.0:
            ggd_score = min(ggd_score + 15, 100)
            reasons.append(f"Elevated noise (std={noise_std:.3f})")
        
        branch_scores["ggd"] = ggd_score
        
        # Branch 2: Local Patch Consistency (Weight: 20%)
        local_score = 0
        if local_analysis["suspicious_ratio"] > 0.4:
            local_score = 100
            reasons.append(f"{local_analysis['suspicious_patches']} suspicious patches ({local_analysis['suspicious_ratio']*100:.0f}%)")
        elif local_analysis["suspicious_ratio"] > 0.25:
            local_score = 60
        elif local_analysis["suspicious_ratio"] > 0.15:
            local_score = 27
        
        if local_analysis["variance"] > 0.3:
            local_score = min(local_score + 15, 100)
            if local_score >= 60:
                reasons.append(f"High patch variance ({local_analysis['variance']:.2f})")
        branch_scores["local"] = local_score
        
        # Branch 3: Texture Consistency (Weight: 15%)
        texture_score = 0
        if texture_analysis["inconsistency"] > 0.5:
            texture_score = 100
            reasons.append(f"High texture inconsistency ({texture_analysis['inconsistency']:.2f})")
        elif texture_analysis["inconsistency"] > 0.3:
            texture_score = 60
        elif texture_analysis["inconsistency"] > 0.2:
            texture_score = 32
        branch_scores["texture"] = texture_score
        
        # Branch 4: PatchCraft SRM Analysis (Weight: 20%)
        # Normalize patchcraft_score (typical range 0-10, high = AI)
        patchcraft_normalized = min(patchcraft_analysis["patchcraft_score"] * 10, 100)
        if patchcraft_normalized > 70:
            reasons.append(f"PatchCraft: High SRM discrepancy ({patchcraft_analysis['patchcraft_score']:.2f})")
        branch_scores["patchcraft"] = patchcraft_normalized
        
        # Branch 5: Spectral Grid Detection (Weight: 15%)
        # Normalize spectral_score (typical range 0-50, high = AI)
        spectral_normalized = min(spectral_analysis["spectral_score"] * 2, 100)
        if spectral_normalized > 50:
            reasons.append(f"Spectral: Grid artifacts detected ({spectral_analysis['spectral_score']:.1f})")
        branch_scores["spectral"] = spectral_normalized
        
        # Branch 6: Kurtosis (Weight: 5%)
        kurtosis_score = 0
        if 2.5 <= kurtosis <= 3.5:
            kurtosis_score = 100
        elif kurtosis > 4.5:
            kurtosis_score = 0
        else:
            kurtosis_score = 50
        branch_scores["kurtosis"] = kurtosis_score
        
        # Branch 7: Chromatic Aberration (Weight: 10%)
        # Real lenses: > 0.3% radial CA increase (natural optical effect)
        # Smartphones: 0.1-0.3% (ISP corrected, slightly negative possible)
        # AI images: < 0.05% (perfectly flat - no lens)
        lca_score = 0
        if lca_analysis["lca_found"]:
            ca_increase = lca_analysis["lca_displacement"]  # Original value with sign
            ca_absolute = abs(ca_increase) if ca_increase is not None else 0.0  # Absolute magnitude
            
            if ca_absolute < 0.05:
                # Extremely flat - likely AI (no physical lens at all)
                lca_score = 100
                reasons.append(f"No chromatic aberration ({ca_increase:.2f}% - too perfect)")
            elif ca_absolute < 0.2:
                # Very low CA - could be smartphone with heavy correction
                lca_score = 40  # Slightly suspicious but could be computational photography
                reasons.append(f"Minimal CA ({ca_increase:.2f}% - smartphone ISP correction?)")
            elif ca_absolute >= 0.2 and ca_absolute <= 3.0:
                # Normal CA range - real camera (DSLR or older smartphone)
                lca_score = 0
                reasons.append(f"Natural lens aberration ({ca_increase:.2f}%)")
            else:
                # Very high CA - might be noise or cheap lens
                lca_score = 20
                reasons.append(f"Unusual CA pattern ({ca_increase:.2f}%)")
        else:
            lca_score = 50  # Unknown
        branch_scores["chromatic"] = lca_score
        
        # === WEIGHTED FUSION ===
        weights = {
            "ggd": 0.20,
            "local": 0.18,
            "texture": 0.12,
            "patchcraft": 0.18,
            "spectral": 0.12,
            "chromatic": 0.10,
            "kurtosis": 0.10
        }
        
        # Calculate weighted average
        ai_score = sum(branch_scores[key] * weights[key] for key in weights.keys())
        
        # === SSP TIEBREAKER ===
        # Use Simplest Smooth Patch as tiebreaker for all scores
        ssp_adjustment = 0
        if ssp_analysis["found_patch"]:
            ssp_beta = ssp_analysis["ssp_beta"]
            
            # Real sensor noise in smooth areas: β ≈ 0.6-0.9
            # AI Gaussian remnants: β ≈ 2.0, or over-smoothed: β > 5.0
            if 1.7 <= ssp_beta <= 2.3:
                # Gaussian-like in smooth area = AI signature
                ssp_adjustment = 20
                ai_score = min(ai_score + ssp_adjustment, 100)
                reasons.append(f"SSP: Gaussian smooth area (β={ssp_beta:.2f})")
            elif ssp_beta > 5.0:
                # Over-smoothed = AI artifact
                ssp_adjustment = 15
                ai_score = min(ai_score + ssp_adjustment, 100)
                reasons.append(f"SSP: Over-smoothed area (β={ssp_beta:.2f})")
            elif ssp_beta < 1.0:
                # Natural sensor noise = likely real (reduce AI score more significantly)
                ssp_adjustment = -20
                ai_score = max(ai_score + ssp_adjustment, 0)
                reasons.append(f"SSP: Natural sensor noise (β={ssp_beta:.2f})")
        
        confidence = min(int(ai_score), 99)
        
        # Determine verdict
        if ai_score >= 70:
            verdict = "ai_generated"
            is_ai = True
            summary = "High probability of AI generation - synthetic noise patterns detected"
        elif ai_score >= 50:
            verdict = "likely_ai"
            is_ai = True
            summary = "Likely AI-generated based on noise distribution analysis"
        elif ai_score >= 30:
            verdict = "uncertain"
            is_ai = None  # Neither true nor false - uncertain
            summary = "Potential AI/Image Alteration - Examples: Nano-edited photos, Banana filter, heavy post-processing, upscaled images"
        else:
            verdict = "likely_real"
            is_ai = False
            summary = "Consistent with natural photography - organic noise patterns"
        
        result = {
            "is_ai_generated": is_ai,
            "confidence": confidence,
            "verdict": verdict,
            "reason": summary,
            "details": " | ".join(reasons[:4]),  # Top 4 reasons
            "metrics": {
                "ggd_shape": float(shape),
                "ggd_scale": float(scale),
                "kurtosis": float(kurtosis),
                "noise_std": float(noise_std),
                "ai_score": int(ai_score),
                "local_suspicious_patches": local_analysis["suspicious_patches"],
                "local_suspicious_ratio": local_analysis["suspicious_ratio"],
                "local_variance": local_analysis["variance"],
                "local_mean_beta": local_analysis["mean_beta"],
                "texture_rich_beta": texture_analysis["rich_beta"],
                "texture_poor_beta": texture_analysis["poor_beta"],
                "texture_inconsistency": texture_analysis["inconsistency"],
                "ssp_beta": ssp_analysis["ssp_beta"],
                "ssp_variance": ssp_analysis["ssp_variance"],
                "ssp_found": ssp_analysis["found_patch"],
                "patchcraft_score": patchcraft_analysis["patchcraft_score"],
                "patchcraft_rich": patchcraft_analysis["rich_feature"],
                "patchcraft_poor": patchcraft_analysis["poor_feature"],
                "spectral_score": spectral_analysis["spectral_score"],
                "spectral_period_4": spectral_analysis["period_4"],
                "spectral_period_8": spectral_analysis["period_8"],
                "spectral_period_16": spectral_analysis["period_16"],
                "lca_displacement": lca_analysis["lca_displacement"],
                "lca_found": lca_analysis["lca_found"],
                "branch_scores": branch_scores,
                "fusion_weights": weights
            },
            "noise_distribution": {
                "histogram": {
                    "bins": bin_centers.tolist(),
                    "values": hist.tolist()
                },
                "ggd_fit": {
                    "x": x_range.tolist(),
                    "y": ggd_curve.tolist()
                },
                "gaussian_ref": {
                    "x": x_range.tolist(),
                    "y": gaussian_curve.tolist()
                }
            },
            "local_consistency": {
                "heatmap": local_analysis["heatmap"],
                "blocks_analyzed": local_analysis["blocks_analyzed"],
                "suspicious_patches": local_analysis["suspicious_patches"],
                "max_beta": local_analysis["max_beta"],
                "min_beta": local_analysis["min_beta"]
            }
        }
        
        # === AI REASONING VERIFICATION ===
        # Prepare metrics for AI reasoning model
        reasoning_input = {
            "total_weighted_score": int(ai_score),
            "statistical_verdict": verdict,
            "branch_1_ggd": {
                "beta_shape": float(shape),
                "noise_std": float(noise_std),
                "kurtosis": float(kurtosis)
            },
            "branch_2_local_patches": {
                "suspicious_ratio": float(local_analysis["suspicious_ratio"]),
                "suspicious_patches": int(local_analysis["suspicious_patches"]),
                "mean_beta": float(local_analysis["mean_beta"])
            },
            "branch_3_texture": {
                "inconsistency": float(texture_analysis["inconsistency"]),
                "rich_beta": float(texture_analysis["rich_beta"]),
                "poor_beta": float(texture_analysis["poor_beta"])
            },
            "branch_4_patchcraft": {
                "srm_score": float(patchcraft_analysis["patchcraft_score"]),
                "rich_feature": float(patchcraft_analysis["rich_feature"]),
                "poor_feature": float(patchcraft_analysis["poor_feature"])
            },
            "branch_5_spectral": {
                "grid_score": float(spectral_analysis["spectral_score"]),
                "period_4": float(spectral_analysis["period_4"]),
                "period_8": float(spectral_analysis["period_8"]),
                "period_16": float(spectral_analysis["period_16"])
            },
            "branch_6_kurtosis": {
                "value": float(kurtosis)
            },
            "branch_7_optics": {
                "ca_percentage": float(lca_analysis["lca_displacement"]),
                "ca_found": bool(lca_analysis["lca_found"])
            },
            "ssp_tiebreaker": {
                "beta": float(ssp_analysis["ssp_beta"]),
                "adjustment": float(ssp_adjustment)
            }
        }
        
        # Get AI reasoning
        ai_reasoning = get_ai_reasoning(reasoning_input)
        
        # Add AI reasoning to result
        result["ai_reasoning"] = ai_reasoning
        
        print(f"Successfully generated result. AI Reasoning: {ai_reasoning['verdict']} ({ai_reasoning['confidence_score']}%)")
        print(f"Local consistency heatmap size: {len(result['local_consistency']['heatmap'])}x{len(result['local_consistency']['heatmap'][0]) if result['local_consistency']['heatmap'] else 0}")
        return result
        
    except Exception as e:
        # Fallback to basic analysis if GGD fails
        import traceback
        print(f"Error in detect_ai_generation: {str(e)}")
        print(traceback.format_exc())
        
        return {
            "is_ai_generated": False,
            "confidence": 0,
            "verdict": "error",
            "reason": f"Analysis error: {str(e)}",
            "details": "Could not perform noise distribution analysis",
            "metrics": {
                "ggd_shape": 0,
                "ggd_scale": 0,
                "kurtosis": 0,
                "noise_std": 0,
                "ai_score": 0,
                "local_suspicious_patches": 0,
                "local_suspicious_ratio": 0,
                "local_variance": 0,
                "local_mean_beta": 0,
                "texture_rich_beta": 0,
                "texture_poor_beta": 0,
                "texture_inconsistency": 0,
                "ssp_beta": 0,
                "ssp_variance": 0,
                "ssp_found": False,
                "ssp_used_tiebreaker": False,
                "patchcraft_score": 0,
                "patchcraft_rich": 0,
                "patchcraft_poor": 0,
                "spectral_score": 0,
                "spectral_period_4": 0,
                "spectral_period_8": 0,
                "spectral_period_16": 0,
                "lca_displacement": 0,
                "lca_found": False,
                "branch_scores": {
                    "ggd": 0,
                    "local": 0,
                    "texture": 0,
                    "patchcraft": 0,
                    "spectral": 0,
                    "chromatic": 0,
                    "kurtosis": 0
                },
                "fusion_weights": {
                    "ggd": 0.20,
                    "local": 0.18,
                    "texture": 0.12,
                    "patchcraft": 0.18,
                    "spectral": 0.12,
                    "chromatic": 0.10,
                    "kurtosis": 0.10
                }
            },
            "noise_distribution": {
                "histogram": {"bins": [], "values": []},
                "ggd_fit": {"x": [], "y": []},
                "gaussian_ref": {"x": [], "y": []}
            },
            "local_consistency": {
                "heatmap": [],
                "blocks_analyzed": 0,
                "suspicious_patches": 0,
                "max_beta": 0,
                "min_beta": 0
            }
        }

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "online", "message": "AI Image Detector API"}


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    image_name: Optional[str] = Form(None),
    user_label: Optional[str] = Form(None)
) -> Dict[str, Any]:
    """
    Analyzes an uploaded image for AI-generation artifacts.
    
    Args:
        file: Uploaded image file
        image_name: User-provided name for the experiment
        user_label: User's ground truth label (e.g., "AI", "Real", "Smartphone")
        
    Returns:
        JSON object with frequency, power, and detection results
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file."
        )
    
    try:
        # Read file contents
        contents = await file.read()
        file_size = len(contents)
        
        # Get image dimensions
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image format")
        height, width = img.shape[:2]
        
        # Perform analysis
        frequency, power = get_azimuthal_average(contents)
        
        # Detect AI generation (pass image bytes for noise analysis)
        detection_result = detect_ai_generation(frequency, power, contents)
        
        # Save to database if name and label provided
        experiment_id = None
        image_path = None
        if image_name and user_label:
            try:
                # Save the image file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_filename = f"{timestamp}_{image_name}"
                # Remove any path separators from filename
                safe_filename = safe_filename.replace("/", "_").replace("\\", "_")
                image_path = UPLOADS_DIR / safe_filename
                
                # Save the uploaded file
                with open(image_path, "wb") as buffer:
                    buffer.write(contents)
                
                experiment_id = database.save_experiment(
                    image_name=image_name,
                    user_label=user_label,
                    file_size=file_size,
                    image_dimensions=(width, height),
                    detection_result=detection_result,
                    image_path=str(image_path)
                )
                print(f"✅ Saved experiment #{experiment_id}: {image_name} (label: {user_label})")
            except Exception as db_error:
                print(f"⚠️ Database error: {db_error}")
                # Don't fail the request if database save fails
        
        return {
            "frequency": frequency,
            "power": power,
            "detection": detection_result,
            "experiment_id": experiment_id
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.get("/experiments")
async def get_experiments() -> Dict[str, Any]:
    """Get all experiments from the database."""
    try:
        experiments = database.get_all_experiments()
        stats = database.get_experiment_statistics()
        return {
            "experiments": experiments,
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: int) -> Dict[str, Any]:
    """Get a specific experiment by ID."""
    try:
        experiment = database.get_experiment_by_id(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return experiment
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/experiments/{experiment_id}/image")
async def get_experiment_image(experiment_id: int):
    """Get the saved image for a specific experiment."""
    try:
        experiment = database.get_experiment_by_id(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        image_path = experiment.get("image_path")
        if not image_path or not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image file not found")
        
        return FileResponse(image_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving image: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/experiments/label/{user_label}")
async def get_experiments_by_label(user_label: str) -> List[Dict]:
    """Get all experiments with a specific label."""
    try:
        experiments = database.get_experiments_by_label(user_label)
        return experiments
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
