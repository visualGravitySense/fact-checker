"""
Detailed image analysis for real/fake detection
"""

import sys
import io
# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from face_real_fake_detector import FaceRealFakeDetector
import cv2
import numpy as np

# Create detector
detector = FaceRealFakeDetector()

# Analyze image
image_path = "613648953_1449572173841888_2396410150948109536_n.jpg"

print("="*60)
print("DETAILED IMAGE ANALYSIS")
print("="*60)
print(f"Image: {image_path}\n")

# Load image
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not load image")
    exit(1)

print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels\n")

# Try with face detection
print("1. Analysis with face detection:")
print("-" * 60)
result_with_face = detector.analyze_image(image_path, use_face_detection=True)
print(f"Result: {'REAL' if result_with_face['is_real'] else 'FAKE'}")
print(f"Confidence: {result_with_face['confidence']:.2%}")

# Analyze entire image
print("\n2. Analysis of entire image:")
print("-" * 60)
result_full = detector.analyze_image(image_path, use_face_detection=False)
print(f"Result: {'REAL' if result_full['is_real'] else 'FAKE'}")
print(f"Confidence: {result_full['confidence']:.2%}")

# Detailed information about covariance matrix
if result_full.get('covariance_matrix'):
    C = np.array(result_full['covariance_matrix'])
    print("\n3. Gradient Covariance Matrix:")
    print("-" * 60)
    print(f"  C = [{C[0,0]:.2f}  {C[0,1]:.2f}]")
    print(f"      [{C[1,0]:.2f}  {C[1,1]:.2f}]")
    
    eigenvals = np.linalg.eigvals(C)
    eigenvals = np.sort(eigenvals)[::-1]
    print(f"\n  Eigenvalues:")
    print(f"    λ₁ = {eigenvals[0]:.2f} (largest)")
    print(f"    λ₂ = {eigenvals[1]:.2f} (smallest)")
    
    # Condition number
    if eigenvals[1] > 0:
        condition_num = eigenvals[0] / eigenvals[1]
        print(f"\n  Condition number: {condition_num:.2f}")
        print(f"    (ratio λ₁/λ₂)")
        if condition_num > 2.0:
            print(f"    → High condition number indicates structured gradients")
            print(f"    → This is characteristic of REAL images")
        else:
            print(f"    → Low condition number may indicate less structured gradients")
    
    # Trace and determinant
    trace = np.trace(C)
    det = np.linalg.det(C)
    print(f"\n  Matrix trace: {trace:.2f}")
    print(f"  Determinant: {det:.2f}")

# Feature analysis
if result_full.get('features'):
    features = result_full['features']
    print("\n4. Extracted Features:")
    print("-" * 60)
    print(f"  Eigenvalues: [{features[0]:.2f}, {features[1]:.2f}]")
    print(f"  Trace: {features[2]:.2f}")
    print(f"  Determinant: {features[3]:.2f}")
    print(f"  Condition number: {features[4]:.2f}")
    print(f"  Mean |Gx|: {features[5]:.2f}")
    print(f"  Mean |Gy|: {features[6]:.2f}")
    print(f"  Standard deviation Gx: {features[7]:.2f}")
    print(f"  Standard deviation Gy: {features[8]:.2f}")
    print(f"  Correlation Gx-Gy: {features[9]:.2f}")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
if result_full['is_real']:
    print("Image classified as REAL")
    print("Gradients show structured patterns,")
    print("characteristic of physical objects and real lighting.")
else:
    print("Image classified as FAKE")
    print("Gradients show less structured patterns,")
    print("which may indicate AI generation or manipulation.")

if result_full.get('note'):
    print(f"\nNote: {result_full['note']}")
print("="*60)
