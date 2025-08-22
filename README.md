# Linear Algebra – Image Manipulation Project

This project provides a Python class **`Image`** for performing a wide range of image manipulation tasks using **OpenCV** and **NumPy**.  
With this class, you can easily load, save, display, resize, rotate, zoom, filter, detect edges, blend images, and add logos or text.  

This README explains installation, usage, and the functionality of each method.  

---

## Table of Contents
- [Linear Algebra – Image Manipulation Project](#linear-algebra--image-manipulation-project)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
      - [Initialization](#initialization)
      - [Saving and Displaying Images](#saving-and-displaying-images)
      - [Resizing and Zooming](#resizing-and-zooming)
      - [Rotating](#rotating)
      - [Edge Detection and Blurring](#edge-detection-and-blurring)
      - [Color Manipulation](#color-manipulation)
      - [Adjusting Brightness and Contrast](#adjusting-brightness-and-contrast)
      - [Blending Images](#blending-images)
      - [Adding Logos and Text](#adding-logos-and-text)
  - [Examples](#examples)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/image-manipulation-project.git
```

Navigate to the project directory:

```bash
cd image-manipulation-project
```
Install dependencies:
```bash
Install the required dependencies:
```
## Usage

#### Initialization

```bash
from image_manipulation import Image

# Create an Image object from a file
img = Image("path/to/your/image.png")
```

#### Saving and Displaying Images

save(path) → saves the current image to the specified path.

display(window_name) → opens the image in a window for viewing.
```bash
img.save("output.png")
img.display("Sample Image")
```

#### Resizing and Zooming

resize(factor) → scales the image by a given factor (e.g., 0.5 for half size).

zoomAt(factor) → zooms into the image around its center.

```bash
img.resize(0.5)   # Shrink
img.zoomAt(2.0)   # Zoom in
```

#### Rotating

rotate(angle) → rotates the image by a given angle in degrees.
```bash
img.rotate(90)  # Rotate 90 degrees clockwise
```

#### Edge Detection and Blurring

edge_detection() → applies Canny edge detection to highlight object boundaries.

blur_by_averaging() → smooths the image using an averaging filter.

stronger_blur(rate) → applies a stronger blur effect with adjustable kernel size.

```bash
img.edge_detection()
img.blur_by_averaging()
img.stronger_blur(15)
```

#### Color Manipulation

invert_color() → inverts all pixel colors (like a photo negative).

gray_scale() → converts the image to grayscale.

```bash
img.invert_color()
img.gray_scale()
```


#### Adjusting Brightness and Contrast

adjust_brightness(beta) → increases/decreases brightness by shifting pixel values.

adjust_contrast(alpha) → enhances or reduces contrast by scaling pixel intensity.

```bash
img.adjust_brightness(beta=50)
img.adjust_contrast(alpha=2.0)
```


#### Blending Images

blend_image(path, blend_rate1, blend_rate2) → merges two images together with custom weights.

```bash
img.blend_image("path/to/another.png", 0.7, 0.3)
```


#### Adding Logos and Text

add_logo(path) → overlays a logo onto the image.

add_text(text, position, font_size, color, thickness) → writes custom text on the image.

```bash
img.add_logo("logo.png")
img.add_text("Hello World", position=(50, 50), font_size=2, color=(255,0,0), thickness=2)
```

## Examples 
```bash
# Example 1: Resize and save
img = Image("test1.png")
img.resize(0.5)
img.save("resized.png")

# Example 2: Rotate and display
img = Image("test1.png")
img.rotate(45)
img.display("Rotated Image")

# Example 3: Edge detection and save
img = Image("test1.png")
img.edge_detection()
img.save("edges.png")
```
