"""
Face Real/Fake Detection using Gradient and Covariance Matrix Analysis

This program implements the method described in task.md to detect if a face
in an image is real or AI-generated/fake by analyzing gradient patterns
and covariance matrices.
"""

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import argparse
from pathlib import Path


class FaceRealFakeDetector:
    """Detector for real vs fake faces using gradient covariance analysis."""
    
    def __init__(self):
        self.classifier = None
        self.scaler = StandardScaler()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def rgb_to_luminance(self, image):
        """
        Convert RGB image to luminance (grayscale).
        
        Formula: L(x,y) = 0.2126*R + 0.7152*G + 0.0722*B
        """
        if len(image.shape) == 3:
            # Convert BGR to RGB if using OpenCV
            if image.shape[2] == 3:
                r, g, b = image[:, :, 2], image[:, :, 1], image[:, :, 0]
            else:
                r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            return luminance.astype(np.float64)
        else:
            return image.astype(np.float64)
    
    def compute_gradients(self, luminance):
        """
        Compute horizontal and vertical gradients.
        
        Returns:
            Gx: horizontal gradient (∂L/∂x)
            Gy: vertical gradient (∂L/∂y)
        """
        # Using Sobel operator for gradient computation
        Gx = cv2.Sobel(luminance, cv2.CV_64F, 1, 0, ksize=3)
        Gy = cv2.Sobel(luminance, cv2.CV_64F, 0, 1, ksize=3)
        
        return Gx, Gy
    
    def compute_covariance_matrix(self, Gx, Gy):
        """
        Compute covariance matrix from gradients.
        
        Steps:
        1. Create matrix M where each row is [Gx(i), Gy(i)] for pixel i
        2. Compute C = (1/N) * M^T * M
        
        Returns:
            covariance_matrix: 2x2 covariance matrix
            features: extracted features from covariance matrix
        """
        # Flatten gradients
        gx_flat = Gx.flatten()
        gy_flat = Gy.flatten()
        
        # Create matrix M: N×2 where N is number of pixels
        M = np.column_stack([gx_flat, gy_flat])
        
        # Remove any NaN or Inf values
        valid_mask = np.isfinite(M).all(axis=1)
        M = M[valid_mask]
        
        if len(M) == 0:
            return None, None
        
        # Compute covariance matrix: C = (1/N) * M^T * M
        N = len(M)
        C = (1.0 / N) * M.T @ M
        
        # Extract features from covariance matrix
        features = self.extract_features(C, M)
        
        return C, features
    
    def extract_features(self, C, M):
        """
        Extract statistical features from covariance matrix and gradient data.
        
        Features include:
        - Eigenvalues of covariance matrix
        - Trace and determinant
        - Gradient statistics
        - Correlation coefficient
        """
        features = []
        
        # Eigenvalues of covariance matrix
        eigenvals = np.linalg.eigvals(C)
        eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
        features.extend(eigenvals)
        
        # Trace (sum of diagonal elements)
        trace = np.trace(C)
        features.append(trace)
        
        # Determinant
        det = np.linalg.det(C)
        features.append(det)
        
        # Condition number (ratio of largest to smallest eigenvalue)
        if eigenvals[1] > 0:
            condition_num = eigenvals[0] / eigenvals[1]
        else:
            condition_num = 0
        features.append(condition_num)
        
        # Gradient statistics
        gx_flat = M[:, 0]
        gy_flat = M[:, 1]
        
        features.append(np.mean(np.abs(gx_flat)))
        features.append(np.mean(np.abs(gy_flat)))
        features.append(np.std(gx_flat))
        features.append(np.std(gy_flat))
        
        # Correlation coefficient
        if np.std(gx_flat) > 0 and np.std(gy_flat) > 0:
            correlation = np.corrcoef(gx_flat, gy_flat)[0, 1]
        else:
            correlation = 0
        features.append(correlation)
        
        # Additional gradient magnitude statistics
        gradient_magnitude = np.sqrt(gx_flat**2 + gy_flat**2)
        features.append(np.mean(gradient_magnitude))
        features.append(np.std(gradient_magnitude))
        features.append(np.percentile(gradient_magnitude, 95))
        
        return np.array(features)
    
    def detect_face(self, image):
        """Detect face in image using Haar Cascade."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces
    
    def analyze_image(self, image_path, use_face_detection=True):
        """
        Analyze an image to determine if it's real or fake.
        
        Args:
            image_path: Path to image file
            use_face_detection: If True, only analyze face region
        
        Returns:
            result: Dictionary with analysis results
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Detect face if requested
        if use_face_detection:
            faces = self.detect_face(image)
            if len(faces) > 0:
                # Use the largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                # Expand face region slightly
                margin = int(min(w, h) * 0.2)
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(image.shape[1] - x, w + 2 * margin)
                h = min(image.shape[0] - y, h + 2 * margin)
                roi = image[y:y+h, x:x+w]
            else:
                print("Warning: No face detected. Analyzing entire image.")
                roi = image
        else:
            roi = image
        
        # Convert to luminance
        luminance = self.rgb_to_luminance(roi)
        
        # Compute gradients
        Gx, Gy = self.compute_gradients(luminance)
        
        # Compute covariance matrix and extract features
        C, features = self.compute_covariance_matrix(Gx, Gy)
        
        if features is None:
            return {
                'is_real': None,
                'confidence': 0.0,
                'features': None,
                'error': 'Could not extract features'
            }
        
        # Predict if classifier is trained
        if self.classifier is not None:
            features_scaled = self.scaler.transform([features])
            prediction = self.classifier.predict(features_scaled)[0]
            probabilities = self.classifier.predict_proba(features_scaled)[0]
            confidence = max(probabilities)
            
            result = {
                'is_real': bool(prediction),
                'confidence': float(confidence),
                'features': features.tolist(),
                'covariance_matrix': C.tolist() if C is not None else None,
                'eigenvalues': np.linalg.eigvals(C).tolist() if C is not None else None
            }
        else:
            # Use heuristic-based classification
            # Real images typically have:
            # - Higher condition number (more structured gradients)
            # - More consistent gradient patterns
            condition_num = features[3] if len(features) > 3 else 0
            correlation = features[9] if len(features) > 9 else 0
            
            # Simple heuristic (can be improved with training data)
            is_real = condition_num > 2.0 and abs(correlation) < 0.5
            
            result = {
                'is_real': is_real,
                'confidence': 0.5,  # Low confidence without training
                'features': features.tolist(),
                'covariance_matrix': C.tolist() if C is not None else None,
                'eigenvalues': np.linalg.eigvals(C).tolist() if C is not None else None,
                'note': 'Using heuristic classification. Train model for better accuracy.'
            }
        
        return result
    
    def train(self, real_images_dir, fake_images_dir):
        """
        Train classifier on labeled data.
        
        Args:
            real_images_dir: Directory containing real face images
            fake_images_dir: Directory containing fake/AI-generated face images
        """
        print("Collecting training data...")
        X = []
        y = []
        
        # Load real images
        real_paths = list(Path(real_images_dir).glob('*.jpg')) + \
                     list(Path(real_images_dir).glob('*.png')) + \
                     list(Path(real_images_dir).glob('*.jpeg'))
        
        for img_path in real_paths:
            try:
                result = self.analyze_image(str(img_path), use_face_detection=True)
                if result['features'] is not None:
                    X.append(result['features'])
                    y.append(1)  # 1 = real
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Load fake images
        fake_paths = list(Path(fake_images_dir).glob('*.jpg')) + \
                     list(Path(fake_images_dir).glob('*.png')) + \
                     list(Path(fake_images_dir).glob('*.jpeg'))
        
        for img_path in fake_paths:
            try:
                result = self.analyze_image(str(img_path), use_face_detection=True)
                if result['features'] is not None:
                    X.append(result['features'])
                    y.append(0)  # 0 = fake
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        if len(X) < 2:
            raise ValueError("Not enough training data. Need at least 2 samples.")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Training on {len(X)} samples ({np.sum(y)} real, {len(y) - np.sum(y)} fake)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.classifier.score(X_train_scaled, y_train)
        test_score = self.classifier.score(X_test_scaled, y_test)
        
        print(f"Training accuracy: {train_score:.2%}")
        print(f"Test accuracy: {test_score:.2%}")
        
        return train_score, test_score


def main():
    parser = argparse.ArgumentParser(
        description='Detect if a face in an image is real or AI-generated/fake'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to image file to analyze'
    )
    parser.add_argument(
        '--no-face-detection',
        action='store_true',
        help='Analyze entire image instead of just face region'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train model on labeled data'
    )
    parser.add_argument(
        '--real-dir',
        type=str,
        help='Directory with real images (for training)'
    )
    parser.add_argument(
        '--fake-dir',
        type=str,
        help='Directory with fake images (for training)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to save/load trained model'
    )
    
    args = parser.parse_args()
    
    detector = FaceRealFakeDetector()
    
    # Train if requested
    if args.train:
        if not args.real_dir or not args.fake_dir:
            print("Error: --real-dir and --fake-dir required for training")
            return
        detector.train(args.real_dir, args.fake_dir)
        print("Training completed!")
        return
    
    # Analyze image
    try:
        result = detector.analyze_image(
            args.image_path,
            use_face_detection=not args.no_face_detection
        )
        
        print("\n" + "="*50)
        print("FACE REAL/FAKE DETECTION RESULTS")
        print("="*50)
        print(f"Image: {args.image_path}")
        print(f"Result: {'REAL' if result['is_real'] else 'FAKE'}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        if 'eigenvalues' in result and result['eigenvalues']:
            print(f"\nCovariance Matrix Eigenvalues: {result['eigenvalues']}")
        
        if 'note' in result:
            print(f"\nNote: {result['note']}")
        
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
