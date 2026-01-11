# Gradient Analysis for Real/Fake Image Detection

This document explains the gradient analysis method used in the Face Real/Fake Detector, as illustrated in `image.png`.

## Overview

Gradient analysis is a fundamental step in distinguishing real images from AI-generated ones. The method is based on the principle that real images reflect physical properties of the environment, while fake images only imitate statistical patterns learned from training data.

## Image Processing Pipeline

The gradient analysis process consists of six main stages, each revealing different aspects of the image structure:

### 1. Original Image (Оригинальное изображение)

The starting point is a color image containing the subject to be analyzed. This can be a face, an object, or any scene. The original image contains full color information (RGB channels) and represents the raw input data.

### 2. Luminance (Яркость)

**Formula:** `L(x,y) = 0.2126 * R(x,y) + 0.7152 * G(x,y) + 0.0722 * B(x,y)`

The first step is to extract only the brightness (luminance) of each pixel, removing all color information. This grayscale representation preserves only the intensity values, which are crucial for gradient computation.

**Why luminance?**
- Brightness gradients are highly sensitive to the physical structure of objects
- They capture how light interacts with surfaces
- Color information can introduce noise that obscures structural patterns

### 3. Horizontal Gradient (Gx = ∂L/∂x)

**Formula:** `Gx(x,y) = ∂L(x,y) / ∂x`

The horizontal gradient measures the rate of change in brightness along the x-axis (left to right). It highlights:
- **Vertical edges**: Sharp transitions that occur horizontally
- **Horizontal structures**: Features that have strong left-right intensity variations
- **Edge orientation**: Primarily detects edges that are vertical or near-vertical

**Visual characteristics:**
- Black areas indicate regions with no horizontal brightness change (smooth horizontal transitions)
- White/light gray areas highlight vertical edges and sharp horizontal transitions
- The intensity represents the magnitude of horizontal change

### 4. Vertical Gradient (Gy = ∂L/∂y)

**Formula:** `Gy(x,y) = ∂L(x,y) / ∂y`

The vertical gradient measures the rate of change in brightness along the y-axis (top to bottom). It highlights:
- **Horizontal edges**: Sharp transitions that occur vertically
- **Vertical structures**: Features that have strong top-bottom intensity variations
- **Edge orientation**: Primarily detects edges that are horizontal or near-horizontal

**Visual characteristics:**
- Black areas indicate regions with no vertical brightness change (smooth vertical transitions)
- White/light gray areas highlight horizontal edges and sharp vertical transitions
- The intensity represents the magnitude of vertical change

### 5. Gradient Magnitude (|∇L|)

**Formula:** `|∇L| = √(Gx² + Gy²)`

The gradient magnitude combines both horizontal and vertical gradient information to show the overall strength of brightness changes at each pixel, regardless of direction.

**Visual characteristics:**
- Uses a heatmap color scheme (dark red → bright yellow)
- Dark red/black areas: regions with minimal brightness change (smooth surfaces)
- Bright yellow areas: regions with strong brightness changes (sharp edges, fine details)
- Provides a comprehensive view of all edges and structural features

**Why it's important:**
- Captures all significant brightness transitions
- Highlights fine surface details, shadows, and structural features
- In real images, these patterns are logically structured and consistent
- In fake images, these patterns may be less organized or contain unnatural artifacts

### 6. Gradient Direction (Color-coded)

**Formula:** `θ = arctan2(Gy, Gx)`

The gradient direction visualizes the orientation of brightness changes using color coding. Each color represents a specific direction of the gradient vector.

**Visual characteristics:**
- Different colors represent different gradient orientations
- Shows the direction in which brightness changes most rapidly
- Helps identify edge orientations and structural patterns
- Black areas indicate regions with minimal gradient (smooth regions)

## Key Differences: Real vs. Fake Images

### Real Images

- **Structured gradients**: Gradients form sharp, consistent edges that align with 3D object structure
- **Logical patterns**: Edge patterns follow physical laws (lighting, shadows, surface geometry)
- **Consistent organization**: Gradient maps show clear, predictable structures
- **Physical accuracy**: Gradients reflect real-world light-surface interactions

### Fake/AI-Generated Images

- **Less organized gradients**: Edges may be less sharp and less consistent
- **Unnatural patterns**: Random or illogical patterns may appear in smooth regions
- **Statistical imitation**: Patterns mimic training data rather than physical reality
- **Inconsistent structure**: Gradient maps may show artifacts or irregularities

## From Gradients to Covariance Matrix

The gradient vectors `[Gx(x,y), Gy(x,y)]` for all pixels are collected into a matrix:

```
M = [ Gx(1)  Gy(1) ]
    [ Gx(2)  Gy(2) ]
    [   :      :   ]
    [ Gx(N)  Gy(N) ]
```

Where `M ∈ R^(N×2)` and N is the total number of pixels.

The covariance matrix is then computed:

**Formula:** `C = (1/N) * M^T * M`

This 2×2 covariance matrix captures:
- **Variance** of horizontal gradients (C[0,0])
- **Variance** of vertical gradients (C[1,1])
- **Covariance** between horizontal and vertical gradients (C[0,1] = C[1,0])

## Statistical Features

From the covariance matrix, we extract features that distinguish real from fake images:

1. **Eigenvalues**: Reveal the principal directions of gradient variation
2. **Trace**: Sum of diagonal elements (total variance)
3. **Determinant**: Product of eigenvalues (measure of spread)
4. **Condition Number**: Ratio of largest to smallest eigenvalue (structure measure)
5. **Correlation**: Relationship between horizontal and vertical gradients

## Practical Application

The `visualize_gradients.py` script generates all these visualizations for any input image, allowing you to:

- Inspect gradient patterns visually
- Compare real vs. fake image characteristics
- Understand why certain images are classified as real or fake
- Debug and improve the detection algorithm

## Example Usage

```bash
python visualize_gradients.py path/to/image.jpg --output analysis
```

This will generate:
- `analysis_visualization.png` - Complete 6-panel visualization
- `analysis_luminance.png` - Luminance map
- `analysis_gradient_x.png` - Horizontal gradients
- `analysis_gradient_y.png` - Vertical gradients
- `analysis_gradient_magnitude.png` - Gradient magnitude

## References

- See `task.md` for the original method description
- See `image.png` for visual examples of each processing stage
- See `face_real_fake_detector.py` for implementation details
