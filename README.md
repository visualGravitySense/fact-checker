# Face Real/Fake Detector

A program to determine whether a face in an image is real or AI-generated (deepfake) using gradient and covariance matrix analysis.

## Method

The program implements the method described in `task.md`:

1. **Luminance Conversion**: The image is converted to grayscale (luminance)
2. **Gradient Computation**: Horizontal (Gx) and vertical (Gy) brightness gradients are computed
3. **Covariance Matrix**: A matrix M is created from all gradient vectors and the covariance matrix C = (1/N) * M^T * M is computed
4. **Feature Extraction**: Statistical features are extracted from the covariance matrix (eigenvalues, trace, determinant, etc.)
5. **Classification**: Machine learning or heuristic methods are used to determine if the image is real

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (without training)

```bash
python face_real_fake_detector.py path/to/image.jpg
```

### Analyze Entire Image (without face detection)

```bash
python face_real_fake_detector.py path/to/image.jpg --no-face-detection
```

### Train Model on Labeled Data

For better accuracy, you can train the model on a dataset:

```bash
python face_real_fake_detector.py dummy.jpg --train --real-dir ./real_faces --fake-dir ./fake_faces
```

After training, the model will use the trained classifier for more accurate predictions.

## Project Structure

- `face_real_fake_detector.py` - main script with `FaceRealFakeDetector` class
- `requirements.txt` - Python dependencies
- `task.md` - method description
- `image.png` - method diagram

## How It Works

1. **Face Detection**: Haar Cascade is used to detect faces in the image
2. **Gradient Analysis**: Real images have more structured and consistent gradients that correspond to the physical structure of objects
3. **Statistical Analysis**: The covariance matrix of gradients reveals patterns that differ between real and generated images
4. **Classification**: Extracted features are used to determine whether the image is real or fake

## Notes

- Without training, the model uses a heuristic method with low confidence
- For best results, it is recommended to train the model on labeled data
- The method works best on face images with good lighting and resolution
