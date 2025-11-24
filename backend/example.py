import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import os

def get_azimuthal_average(image_path, label=None):
    """
    Calculates the Azimuthal Average Power Spectrum of an image.
    Essentially: "How much energy is there at every frequency ring?"
    """
    # 1. Load Image in Grayscale (We care about luminance physics, not color)
    img = cv2.imread(image_path, 0)
    
    if img is None:
        print(f"Error: Could not load {image_path}")
        return None, None

    # 2. FFT (Fast Fourier Transform) -> Move from Spatial to Frequency Domain
    f = fftpack.fft2(img)
    fshift = fftpack.fftshift(f) # Shift zero freq to center
    
    # Calculate Power Spectrum (Log scale to see details)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

    # 3. Compute Azimuthal Average (Radial Profile)
    # We create a grid of radii (r) from the center
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    r = np.sqrt(x**2 + y**2).astype(int)

    # Bin the energy values by their radius (r)
    # This averages all pixels that are "x" distance from the center
    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    
    # Avoid division by zero
    radial_profile = tbin / (nr + 1e-8)

    # Normalize X-axis (0 to 1) so we can compare different image sizes
    normalized_freq = np.linspace(0, 1, len(radial_profile))
    
    return normalized_freq, radial_profile

def run_experiment(real_img_path, ai_img_path):
    print(f"Analyzing Real: {real_img_path}")
    freq_real, profile_real = get_azimuthal_average(real_img_path)

    print(f"Analyzing AI:   {ai_img_path}")
    freq_ai, profile_ai = get_azimuthal_average(ai_img_path)

    if freq_real is None or freq_ai is None:
        return

    # --- PLOTTING THE FORENSICS ---
    plt.figure(figsize=(12, 6))
    
    # Plot curves
    plt.plot(freq_real, profile_real, label='Real Photo', color='blue', linewidth=2, alpha=0.8)
    plt.plot(freq_ai, profile_ai, label='AI Generated', color='red', linewidth=2, alpha=0.8, linestyle='--')

    # Add theoretical "Natural Image" slope (Optional visualization of 1/f law)
    # Natural images usually drop linearly on this log-plot
    
    plt.title("Fundamental Frequency Analysis: Real vs AI")
    plt.xlabel("Spatial Frequency (0 = Low Detail, 1 = Nyquist/Noise)")
    plt.ylabel("Power Spectrum Energy (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Highlight the "High Frequency" tail where AI usually fails
    plt.axvspan(0.7, 1.0, color='yellow', alpha=0.1, label='High Freq Artifact Zone')
    
    plt.show()

# --- INPUT YOUR IMAGES HERE ---
# Replace these with your actual file paths
real_photo = "path/to/your/real_photo.jpg" 
ai_photo = "path/to/your/midjourney_image.png"

# Create dummy files if they don't exist just to test the script logic
if not os.path.exists(real_photo):
    print("⚠️ Please edit the script with paths to your actual images!")
    print("Creating dummy noise images for demonstration...")
    cv2.imwrite("dummy_real.jpg", np.random.randint(0, 255, (512, 512), dtype=np.uint8))
    cv2.imwrite("dummy_ai.jpg", np.random.randint(0, 255, (512, 512), dtype=np.uint8))
    real_photo = "dummy_real.jpg"
    ai_photo = "dummy_ai.jpg"

run_experiment(real_photo, ai_photo)