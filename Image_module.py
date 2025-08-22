import string
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class Image:
    def __init__(self, file_path: string) -> None:
        """
        Initialize an Image object with the image located at the specified file path.

        Parameters:
            file_path (str): The path to the input image file.
        """
        self.image = cv2.imread(file_path)
        self.image_height, self.image_width, self.image_channel = self.image.shape
        self.image_b, self.image_g, self.image_r = cv2.split(self.image )
        
    def save(self, name: str) -> None:
        """
        Save the image with the given name.

        Parameters:
            name (str): The name of the saving image file.
        """
        
        new_folder = "output_images"
        os.makedirs(new_folder, exist_ok=True)
        
        file_path = os.path.join(new_folder, name)
        cv2.imwrite(file_path, self.image)

    def display(self, title: str) -> None:
        """
        Display the image with the given title.

        Parameters:
            title (str): The title to be displayed with the image.
        """
        
        plt.imshow(self.image)
        plt.title(title)
        plt.show()  
        
    def resize(self, resize_factor: float) -> None:
        """
        Resize the image by the specified resize factor.

        Parameters:
            resize_factor (float): The factor by which to resize the image.
        """
        
        new_horizental = self.image_width * resize_factor
        new_vertical = self.image_height * resize_factor
        
        new_image = cv2.resize(self.image, (new_horizental, new_vertical), interpolation = cv2.INTER_CUBIC)
        self.image = new_image

    def zoomAt(self, zoom_factor: float, coords: list = None) -> None:
        """
        Zoom the image at the specified zoom factor around the given coordinates.

        Parameters:
            zoom_factor (float): The factor by which to zoom the image.
            coords (list, optional): The coordinates around which to zoom the image.
        """
        
        if coords is None:
            centerX = (self.image_width - 1) // 2
            centerY = (self.image_height - 1) // 2
        else:
            centerX, centerY = [zoom_factor * i for i in coords]
            
        new_image = cv2.resize(self.image, (0, 0), fx=zoom_factor, fy=zoom_factor, interpolation = cv2.INTER_CUBIC)
        
        
        cropY_low_bound = int(centerY - self.image_height / zoom_factor * 0.5)
        cropY_high_bound = int(centerY + self.image_height / zoom_factor * 0.5)
        
        cropX_low_bound = int(centerX - self.image_width / zoom_factor * 0.5)
        cropX_high_bound = int(centerX + self.image_width / zoom_factor * 0.5)
        
        new_image = new_image[cropY_low_bound : cropY_high_bound, cropX_low_bound : cropX_high_bound]
        self.image = new_image
         
    def rotate(self, degree: float) -> None:
        """
        Rotate the image by the specified degree.

        Parameters:
            degree (float): The angle of rotation in degrees.
        """
        
        centerX = (self.image_width - 1) // 2
        centerY = (self.image_height - 1) // 2
        
        M = cv2.getRotationMatrix2D((centerX, centerY), degree, 1)
        
        new_image = cv2.warpAffine(self.image, M, (self.width, self.height))
        self.image = new_image
             
    def edge_detection(self, low_threshhold: int = 50, high_threshhold: int = 100) -> None:
        """
        Perform edge detection on the image using the Canny edge detection algorithm.

        Parameters:
            low_threshhold (int, optional): The low threshold value for edge detection. Defaults to 50.
            high_threshhold (int, optional): The high threshold value for edge detection. Defaults to 100.
        """
           
        grayscale_img = self.gray_scale()
        
        new_image = cv2.Canny(grayscale_img, low_threshhold, high_threshhold)
        self.image = new_image
 
    def blur_by_averaging(self) -> None:
        """
        Blur the image using averaging filter.
        """
        
        kernel = np.ones((5,5),np.float32) / 25
        
        new_image = cv2.filter2D(self.image, -1, kernel)
        self.image = new_image
        
    def stronger_blur(self, blur_rate: int = 10) -> None:
        """
        Blur the image using built-in OpenCV blur function.

        Parameters:
            blur_rate (int, optional): The blur rate. Defaults to 10.
        """
        
        new_image = cv2.blur(self.image, (blur_rate, blur_rate))
        self.image = new_image
                 
    def sharpen(self) -> None:
        """
        Sharpen the image.
        """
        
        kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
        
        new_image = cv2.filter2D(self.image, -1, kernel)
        self.image = new_image
        
    def reflect(self, reflect_rate: int = 10) -> None:
        """
        Reflect the image.

        Parameters:
            reflect_rate (int, optional): The reflection rate. Defaults to 10.
        """
        
        new_image = cv2.copyMakeBorder(self.image, reflect_rate, reflect_rate, reflect_rate, reflect_rate, reflect_rate)
        self.image = new_image
    
    def invert_color(self) -> None:
        """
        Invert the colors of the image.
        """
        
        new_image = 255 - self.image
        self.image = new_image
        
    def gray_scale(self) -> None:
        """
        Convert the image to grayscale.
        """
        
        new_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = new_image
    
    def adjust_brightness(self, beta: int = 50) -> None:
        """
        Adjust the brightness of the image.

        Parameters:
            beta (int, optional): The brightness adjustment value. Defaults to 50.
        """
        
        new_image = np.zeros(self.image.shape, self.image.dtype)
        
        
        for y in range(self.image_height):
            for x in range(self.image_width):
                for c in range(self.image_channel):
                    new_image[y,x,c] = np.clip(self.image[y,x,c] + beta, 0, 255)
                    
    def adjust_contrast(self, alpha: float = 2.0) -> None:
        """
        Adjust the contrast of the image.

        Parameters:
            alpha (float, optional): The contrast adjustment factor. Defaults to 2.0.
        """
        
        if alpha < 1.0 or alpha > 3.0:
            raise Exception("Error, not in the range")
        
        new_image = np.zeros(self.image.shape, self.image.dtype)
        
        
        for y in range(self.image_height):
            for x in range(self.image_width):
                for c in range(self.image_channel):
                    new_image[y,x,c] = np.clip(alpha * self.image[y,x,c], 0, 255) 

    def blend_image(self, file_path: str, blend_rate1: float = 0.5, blend_rate2: float = 0.5) -> None:
        """
        Blend the image with another image.

        Parameters:
            file_path (str): The path to the image to be blended.
            blend_rate1 (float, optional): The blend rate for the first image. Defaults to 0.5.
            blend_rate2 (float, optional): The blend rate for the second image. Defaults to 0.5.
        """
        
        img2 = cv2.imread(file_path)
        
        new_image = cv2.addWeighted(self.image, blend_rate1, img2, 1 - blend_rate2, 0)
        self.image = new_image
    
    def add_logo(self, file_path: str) -> None:
        """
        Add a logo to the image.

        Parameters:
            file_path (str): The path to the logo image.
        """
        
        logo = cv2.imread(file_path)
        
        rows,cols,channels = logo.shape
        roi = self.image[0:rows, 0:cols]
        
        logo_gray = cv2.cvtColor(logo,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(logo_gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        img2_fg = cv2.bitwise_and(logo,logo,mask = mask)
        
        dst = cv2.add(img1_bg,img2_fg)
        self.image[0:rows, 0:cols ] = dst
        
    def add_text(self, text: str, position = None, font_size :int = 1, color = (0, 255, 0), thickness = 3) -> None:
        """
        Add text to the image.

        Parameters:
            text (str): The text to be added to the image.
            position (tuple, optional): The position where the text will be added. Defaults to the center of the image.
            font_size (int, optional): The font size of the text. Defaults to 1.
            color (tuple, optional): The color of the text. Defaults to (0, 255, 0) (green).
            thickness (int, optional): The thickness of the text. Defaults to 3.
        """
        
        if position is None:
            centerX = (self.image_width - 1) // 2
            centerY = (self.image_height - 1) // 2
            position = (centerX, centerY)
        
        new_image = cv2.putText(self.image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
        self.image = new_image
