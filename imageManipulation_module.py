from hmac import new
import string
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class Image:
    def __init__(self, file_path: string) -> None:
        
        self.image = cv2.imread('test1.png')
        self.image_height, self.image_width, self.image_channel = self.image.shape
        self.image_b, self.image_g, self.image_r = cv2.split(self.image )
        
    def save(self, name: str):
        
        new_folder = "output_images"
        os.makedirs(new_folder, exist_ok=True)
        
        file_path = os.path.join(new_folder, name)
        cv2.imwrite(file_path, self.image)

    def display(self, title: str):
        
        plt.imshow(self.image)
        plt.title(title)
        plt.show()  
    
    #---------------RESIZE
    
    def resize(self, resize_factor: float) -> None:
        
        new_horizental = self.image_width * resize_factor
        new_vertical = self.image_height * resize_factor
        
        new_image = cv2.resize(self.image, (new_horizental, new_vertical), interpolation = cv2.INTER_CUBIC)
        self.image = new_image

        
    def improve_qulaity(self):
        pass
    
    def zoomAt(self, zoom_factor: float, coords: list = None) -> None:
        
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
        
    
    def rotate(self, degree: float):
        
        centerX = (self.image_width - 1) // 2
        centerY = (self.image_height - 1) // 2
        
        M = cv2.getRotationMatrix2D((centerX, centerY), degree, 1)
        
        new_image = cv2.warpAffine(self.image, M, (self.width, self.height))
        self.image = new_image
        
    #----------------FILTER
     
    def edge_detection(self, low_threshhold: int = 50, high_threshhold: int = 100) -> None:
        
        # low_threshhold: incresing this value can help improve the quality and robustness
        #            of the edge detection by reducing noise and enhancing edge localization
        
        # high_threshhold: incresing this value can helpo refine the edge detection process 
        #            by focusing on stronger and more significant edges while suppressing weaker or noisy edges.
        
        grayscale_img = self.gray_scale()
        
        new_image = cv2.Canny(grayscale_img, low_threshhold, high_threshhold)
        self.image = new_image

        
    
    def blur_by_averaging(self) -> None:
        
        kernel = np.ones((5,5),np.float32) / 25
        
        new_image = cv2.filter2D(self.image, -1, kernel)
        self.image = new_image
        
    def blur_built_in(self, blur_rate: int = 10) -> None:
        
        new_image = cv2.blur(self.image, (blur_rate, blur_rate))
        self.image = new_image
        
            
    def sharpen(self):
        
        kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
        
        new_image = cv2.filter2D(self.image, -1, kernel)
        self.image = new_image
        
    def reflect(self, reflect_rate: int = 10) -> None:
        
        new_image = cv2.copyMakeBorder(self.image, reflect_rate, reflect_rate, reflect_rate, reflect_rate, reflect_rate)
        self.image = new_image
    
    
    #---------------COLOR MANIPULATION
    
    # In the context of digital images with 8-bit color depth per channel (such as typical RGB images),
    # the intensity values of each color channel are typically represented using integers ranging from 0 to 255.

    # 0 represents the minimum intensity (black), and
    # 255 represents the maximum intensity (white).
    # This range corresponds to the range of values that can be represented using 8 bits (2^8 = 256 possible values, ranging from 0 to 255).
    # So, 255 is an important number in this context because it represents the maximum intensity value
    # that can be represented in an 8-bit image, and subtracting each pixel value from 255 is a simple
    # and effective way to perform color inversion in such images.
    
    # In RGB color images, each pixel has three color channels: red, green, and blue. Each channel typically uses 8 bits to represent intensity
    # values, resulting in a total of 256 possible intensity levels (0 to 255) for each color channel.
    
    def invert_color(self):
        
        new_image = 255 - self.image
        self.image = new_image
        
    def gray_scale(self):
        
        new_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = new_image

    
    def color_balance(self):
        pass
    
    #in an RGB image, you can adjust the brightness by scaling each of the red, green, and blue channels uniformly.
    def adjust_brightness(self, beta: int = 50) -> None:
        
        new_image = np.zeros(self.image.shape, self.image.dtype)
        
        
        for y in range(self.image_height):
            for x in range(self.image_width):
                for c in range(self.image_channel):
                    new_image[y,x,c] = np.clip(self.image[y,x,c] + beta, 0, 255)
                    
    
    def adjust_contrast(self, alpha: float = 2.0) -> None:
        
        if alpha < 1.0 or alpha > 3.0:
            raise Exception("Error, not in the range")
        
        new_image = np.zeros(self.image.shape, self.image.dtype)
        
        
        for y in range(self.image_height):
            for x in range(self.image_width):
                for c in range(self.image_channel):
                    new_image[y,x,c] = np.clip(alpha * self.image[y,x,c], 0, 255) # what happens if i apply this to all 3 b, g and r channels?

        
    #-----------------ADD ANOTHER IMAGE
    
    def blend_image(self, file_path: str, blend_rate1: float = 0.5, blend_rate2: float = 0.5) -> None:
        
        img2 = cv2.imread(file_path)
        
        new_image = cv2.addWeighted(self.image, blend_rate1, img2, 1 - blend_rate2, 0)
        self.image = new_image
    
    def add_logo(self, file_path: str) -> None:
        
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
        
        if position is None:
            centerX = (self.image_width - 1) // 2
            centerY = (self.image_height - 1) // 2
            position = (centerX, centerY)
        
        new_image = cv2.putText(self.image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
        self.image = new_image
        
        
        
    
