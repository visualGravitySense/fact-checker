"""
Gradient visualization for real/fake image analysis
"""

import sys
import io
# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from face_real_fake_detector import FaceRealFakeDetector

def visualize_gradients(image_path, output_prefix="gradient_analysis", show_plot=True):
    """
    Creates gradient visualization of an image.
    
    Args:
        image_path: Path to image file
        output_prefix: Prefix for output files
    """
    print("Loading image...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    detector = FaceRealFakeDetector()
    
    # Try to find face
    faces = detector.detect_face(image)
    if len(faces) > 0:
        print(f"Faces detected: {len(faces)}")
        # Use largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        margin = int(min(w, h) * 0.2)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        roi = image[y:y+h, x:x+w]
        print(f"Face region: {w}x{h} pixels")
    else:
        print("No face detected, analyzing entire image")
        roi = image
    
    # Convert to luminance
    print("Computing luminance...")
    luminance = detector.rgb_to_luminance(roi)
    
    # Compute gradients
    print("Computing gradients...")
    Gx, Gy = detector.compute_gradients(luminance)
    
    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(Gx**2 + Gy**2)
    
    # Normalize for visualization
    def normalize_for_display(img):
        """Normalizes image for display (0-255)"""
        img_abs = np.abs(img)
        if img_abs.max() > 0:
            normalized = (img_abs / img_abs.max() * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(img, dtype=np.uint8)
        return normalized
    
    Gx_display = normalize_for_display(Gx)
    Gy_display = normalize_for_display(Gy)
    magnitude_display = normalize_for_display(gradient_magnitude)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Original image
    ax1 = plt.subplot(2, 3, 1)
    if len(roi.shape) == 3:
        display_img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    else:
        display_img = roi
    ax1.imshow(display_img)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Luminance
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(luminance, cmap='gray')
    ax2.set_title('Luminance', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. Horizontal gradient (Gx)
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(Gx_display, cmap='gray')
    ax3.set_title('Horizontal Gradient (Gx = ∂L/∂x)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # 4. Vertical gradient (Gy)
    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(Gy_display, cmap='gray')
    ax4.set_title('Vertical Gradient (Gy = ∂L/∂y)', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # 5. Gradient magnitude
    ax5 = plt.subplot(2, 3, 5)
    im5 = ax5.imshow(magnitude_display, cmap='hot')
    ax5.set_title('Gradient Magnitude (|∇L|)', fontsize=12, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    # 6. Combined visualization (color-coded direction)
    ax6 = plt.subplot(2, 3, 6)
    # Compute gradient direction
    gradient_direction = np.arctan2(Gy, Gx)
    # Normalize direction for visualization
    direction_normalized = (gradient_direction + np.pi) / (2 * np.pi)  # 0-1
    magnitude_normalized = gradient_magnitude / gradient_magnitude.max() if gradient_magnitude.max() > 0 else gradient_magnitude
    
    # Create HSV image: Hue = direction, Value = magnitude
    hsv = np.zeros((*gradient_direction.shape, 3))
    hsv[:, :, 0] = direction_normalized  # Hue (direction)
    hsv[:, :, 1] = 1.0  # Saturation
    hsv[:, :, 2] = magnitude_normalized  # Value (magnitude)
    
    rgb_direction = cm.hsv(hsv[:, :, 0])[:, :, :3]
    rgb_direction = rgb_direction * magnitude_normalized[:, :, np.newaxis]
    
    ax6.imshow(rgb_direction)
    ax6.set_title('Gradient Direction (color)', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_file = f"{output_prefix}_visualization.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {output_file}")
    
    # Create separate files for each gradient
    cv2.imwrite(f"{output_prefix}_luminance.png", luminance.astype(np.uint8))
    cv2.imwrite(f"{output_prefix}_gradient_x.png", Gx_display)
    cv2.imwrite(f"{output_prefix}_gradient_y.png", Gy_display)
    cv2.imwrite(f"{output_prefix}_gradient_magnitude.png", magnitude_display)
    
    print(f"Separate images saved:")
    print(f"  - {output_prefix}_luminance.png")
    print(f"  - {output_prefix}_gradient_x.png")
    print(f"  - {output_prefix}_gradient_y.png")
    print(f"  - {output_prefix}_gradient_magnitude.png")
    
    # Statistics
    print("\n" + "="*60)
    print("GRADIENT STATISTICS:")
    print("="*60)
    print(f"Horizontal gradient (Gx):")
    print(f"  Mean: {np.mean(Gx):.2f}")
    print(f"  Standard deviation: {np.std(Gx):.2f}")
    print(f"  Min/Max: {np.min(Gx):.2f} / {np.max(Gx):.2f}")
    print(f"\nVertical gradient (Gy):")
    print(f"  Mean: {np.mean(Gy):.2f}")
    print(f"  Standard deviation: {np.std(Gy):.2f}")
    print(f"  Min/Max: {np.min(Gy):.2f} / {np.max(Gy):.2f}")
    print(f"\nGradient magnitude:")
    print(f"  Mean: {np.mean(gradient_magnitude):.2f}")
    print(f"  Standard deviation: {np.std(gradient_magnitude):.2f}")
    print(f"  Min/Max: {np.min(gradient_magnitude):.2f} / {np.max(gradient_magnitude):.2f}")
    
    # Compute covariance matrix for additional information
    C, features = detector.compute_covariance_matrix(Gx, Gy)
    if C is not None:
        print(f"\nCovariance matrix:")
        print(f"  [{C[0,0]:.2f}  {C[0,1]:.2f}]")
        print(f"  [{C[1,0]:.2f}  {C[1,1]:.2f}]")
        eigenvals = np.linalg.eigvals(C)
        eigenvals = np.sort(eigenvals)[::-1]
        print(f"  Eigenvalues: [{eigenvals[0]:.2f}, {eigenvals[1]:.2f}]")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Gradient visualization for images')
    parser.add_argument('image_path', type=str, help='Path to image file')
    parser.add_argument('--output', type=str, default='gradient_analysis', 
                       help='Prefix for output files')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not show window, only save files')
    
    args = parser.parse_args()
    
    visualize_gradients(args.image_path, args.output, show_plot=not args.no_show)
