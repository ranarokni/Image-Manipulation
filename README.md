# Linear-Algebra-Course
Image Manipulation Project
This project provides a Python class Image for various image manipulation tasks using OpenCV and NumPy. The class includes methods for loading, saving, displaying, resizing, rotating, zooming, applying filters, detecting edges, blending images, and adding logos or text. This README will guide you through the usage of the class and its methods.

Table of Contents
Installation
Usage
Initialization
Saving and Displaying Images
Resizing and Zooming
Rotating
Edge Detection and Blurring
Color Manipulation
Adjusting Brightness and Contrast
Blending Images
Adding Logos and Text
Examples
Contributing
License
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/image-manipulation-project.git
Navigate to the project directory:

bash
Copy code
cd image-manipulation-project
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Initialization
python
Copy code
from image_manipulation import Image

# Initialize an Image object with the path to your image file
img = Image('path/to/your/image.png')
Saving and Displaying Images
python
Copy code
# Save the image
img.save('output_image.png')

# Display the image
img.display('Sample Image')
Resizing and Zooming
python
Copy code
# Resize the image by a factor of 0.5
img.resize(0.5)

# Zoom the image by a factor of 2 around the center
img.zoomAt(2.0)
Rotating
python
Copy code
# Rotate the image by 90 degrees
img.rotate(90)
Edge Detection and Blurring
python
Copy code
# Perform edge detection with default thresholds
img.edge_detection()

# Apply average blurring
img.blur_by_averaging()

# Apply stronger blur with a specified rate
img.stronger_blur(15)
Color Manipulation
python
Copy code
# Invert the colors of the image
img.invert_color()

# Convert the image to grayscale
img.gray_scale()
Adjusting Brightness and Contrast
python
Copy code
# Adjust the brightness of the image
img.adjust_brightness(beta=50)

# Adjust the contrast of the image
img.adjust_contrast(alpha=2.0)
Blending Images
python
Copy code
# Blend the image with another image
img.blend_image('path/to/another/image.png', blend_rate1=0.7, blend_rate2=0.3)
Adding Logos and Text
python
Copy code
# Add a logo to the image
img.add_logo('path/to/logo.png')

# Add text to the image
img.add_text('Sample Text', position=(50, 50), font_size=2, color=(255, 0, 0), thickness=2)
Examples
Here are a few examples of how to use the Image class for various image manipulations:

python
Copy code
# Example 1: Resize and save an image
img = Image('test1.png')
img.resize(0.5)
img.save('resized_image.png')

# Example 2: Rotate and display an image
img = Image('test1.png')
img.rotate(45)
img.display('Rotated Image')

# Example 3: Detect edges and save the image
img = Image('test1.png')
img.edge_detection()
img.save('edges_image.png')
Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any improvements, bug fixes, or feature requests.

Fork the repository
Create your feature branch (git checkout -b feature/my-feature)
Commit your changes (git commit -am 'Add some feature')
Push to the branch (git push origin feature/my-feature)
Create a new Pull Request
License
This project is licensed under the MIT License - see the LICENSE file for details.