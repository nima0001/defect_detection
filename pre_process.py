import cv2 
import numpy as np 
import matplotlib.pyplot as plt

def pre_process(img, side = 0): 
    """
    Process an input image and filter specific HSV values to create a binary mask.
    
    The function converts the input image to the HSV color space and then applies 
    a binary mask based on specified HSV values. Morphological opening is then 
    applied to the mask to remove small noise.

    Parameters:
    - img (numpy.ndarray): The input image in BGR format.
    - side (int, optional): Determines the range of HSV values to filter.
        0: front side 
        1: back side

        
    Returns:
    - mask (numpy.ndarray): A binary mask highlighting regions in the input image 
        that fall within the specified HSV range after morphological opening.
    """
    
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if side == 0: 
        #Lower and upper range of HSV values to filter 
        lower_blue = np.array([67, 0, 90])
        upper_blue = np.array([119, 255, 255])
    else:
        #Lower and upper range of HSV values to filter 
        lower_blue = np.array([ 89,   0, 113])  
        upper_blue = np.array([123, 255, 255])
        
    kernel = np.ones((2,2),np.uint8)  # 2x2 kernel for opening, you can adjust this size if needed

    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask
    

def main():
    img = cv2.imread('front/product_only/M0_L1.bmp')
    mask = pre_process(img)
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()