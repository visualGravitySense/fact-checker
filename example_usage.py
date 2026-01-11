"""
Example usage of FaceRealFakeDetector
"""

from face_real_fake_detector import FaceRealFakeDetector
import cv2

# Create detector
detector = FaceRealFakeDetector()

# Analyze image
image_path = "613648953_1449572173841888_2396410150948109536_n.jpg"

print("Analyzing image...")
result = detector.analyze_image(image_path, use_face_detection=True)

# Display results
print("\nAnalysis results:")
print(f"Image: {image_path}")
print(f"Result: {'REAL' if result['is_real'] else 'FAKE'}")
print(f"Confidence: {result['confidence']:.2%}")

if result.get('eigenvalues'):
    print(f"\nCovariance matrix eigenvalues:")
    for i, val in enumerate(result['eigenvalues'], 1):
        print(f"  λ{i} = {val:.2f}")

if result.get('covariance_matrix'):
    print(f"\nCovariance matrix:")
    C = result['covariance_matrix']
    print(f"  [{C[0][0]:.2f}  {C[0][1]:.2f}]")
    print(f"  [{C[1][0]:.2f}  {C[1][1]:.2f}]")

# Visualization (optional)
image = cv2.imread(image_path)
if image is not None:
    # Show image with result
    faces = detector.detect_face(image)
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            color = (0, 255, 0) if result['is_real'] else (0, 0, 255)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            label = "REAL" if result['is_real'] else "FAKE"
            cv2.putText(image, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Save result
    output_path = "result_" + image_path
    cv2.imwrite(output_path, image)
    print(f"\nResult saved to: {output_path}")
