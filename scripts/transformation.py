"""
Image Transformation for Defect Detection

This module provides functionality to transform a model image to have the same orientation and scale as an instance image
based on two distinct circles present in each image. The alignment is achieved by comparing the line formed by the centers 
of the two circles. Once aligned, the difference between the two images can be used to detect defects.

Functions:
- detect_circles(img): Detects two dominant circles in the input image using Hough transform and returns their centers.
- compute_transformation(circles_A, circles_B): Computes the scale, rotation, and translation needed to align two sets 
  of circle centers.
- apply_transformation(image, scale, rotation, translation, rotation_center): Applies the computed transformation to the 
  input image.
- get_transformed_img1(img1, img2): Transforms the first image based on the orientation and scale of the second image using
  the detected circles.
- get_transformed_bw_img(bw_image, scale, rotation, translation, rotation_center): Applies a computed transformation to a 
  black and white image.

Usage:
Run the module to transform and display the images using the main function. Paths to the images are hardcoded in the main 
function and need to be changed as per the dataset location.

Note:
Ensure OpenCV and NumPy libraries are installed for this module to function correctly.
"""




import cv2
import matplotlib.pyplot as plt
import numpy as np

def detect_circles(img):
    """
    Detect the centers of two dominant circles in the given image.
    
    Parameters:
        img (ndarray): Input image (3-channel).
    
    Returns:
        ndarray: 2x2 array containing the x, y coordinates of the two circle centers.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #canny edge is easier with grayscale image

    img_color = img.copy() #For anotation purpose
    
    
    # Edge detection
    edges = cv2.Canny(gray, 100, 200) #Parameters are Thresholds for hysteris 
    
    # Detect circles using Hough transform
    circles = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, 1,
        minDist = 900, param1=50, param2=20, 
        minRadius= 35, maxRadius=90
        )

    if circles is None:
        return None

    else:
        circles = np.uint16(np.around(circles))[0]
        i = 1
        for circle in circles: 
            x,y,r = circle
            cv2.circle(img_color, center=(x,y), radius= r, color = (0,255,0),thickness = 5)
            print("Circle No.", i,"Coordinate : ", (x,y), "Radius: ", r)
            i+=1
        
        # cv2.line(img_color, circles[0][0:2], circles[1][0:2], (0,0,255), thickness= 3)
        # cv2.imshow("Detected Circles", img_color)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    
    return circles[:,:2]
    
def compute_transformation(circles_A, circles_B):
    """
    Compute the transformation (scale, rotation, translation) needed to align the centers of two sets of circles.
    
    Parameters:
        circles_A (ndarray): 2x2 array of circle centers from the first image.
        circles_B (ndarray): 2x2 array of circle centers from the second image.
        
    Returns:
        tuple: Scale factor, rotation angle (in radians), and translation vector (dx, dy).
    """

    # Extracting circle centers
    c1_A, c2_A = circles_A
    c1_B, c2_B = circles_B
    
    # Sorting circles to make sure we are comparing the correct circles
    c1_A, c2_A = sorted(circles_A, key=lambda x: x[0])
    c1_B, c2_B = sorted(circles_B, key=lambda x: x[0])
    
    # Computing the distance between circles
    dist_A = np.linalg.norm(np.array(c1_A, dtype=np.float64) - np.array(c2_A, dtype=np.float64))
    dist_B = np.linalg.norm(np.array(c1_B, dtype=np.float64) - np.array(c2_B, dtype=np.float64))

    # Scale
    scale = dist_B / dist_A
    
    # Compute rotation angle
    dA = np.array(c2_A, dtype=np.float64) - np.array(c1_A, dtype=np.float64)
    dB = np.array(c2_B, dtype=np.float64) - np.array(c1_B, dtype=np.float64)
    angle_A = np.arctan2(dA[1], dA[0])
    angle_B = np.arctan2(dB[1], dB[0])
    rotation =  angle_A - angle_B 


    # Compute translation
    translation = (c1_B[0] - c1_A[0] * scale, c1_B[1] - c1_A[1] * scale)
    
    return scale, rotation, translation


def apply_transformation(image, scale, rotation, translation, rotation_center):
    """
    Apply the given transformation to the input image.
    
    Parameters:
        image (ndarray): The image to be transformed.
        scale (float): Scaling factor.
        rotation (float): Rotation angle in radians.
        translation (tuple): Translation vector (dx, dy).
        rotation_center (tuple): Center point for rotation (x, y).
    
    Returns:
        ndarray: Transformed image.
    """
    # Get image shape
    rows, cols = image.shape[:2]
    
    # Create the scaling and rotation matrix
    M = cv2.getRotationMatrix2D(rotation_center, np.degrees(rotation), scale)
    
    # Add translation to the transformation matrix
    M[:, 2] += translation
    
    # Apply transformation
    transformed_image = cv2.warpAffine(image, M, (cols, rows))
    
    return transformed_image

def get_transformed_img1(img1, img2):
    """
    Transform the first image to have the same orientation and scale as the second image.
    
    The transformation is computed based on two circles detected in each image.
    
    Parameters:
        img1 (ndarray): The reference image to be transformed.
        img2 (ndarray): The target image that provides the desired orientation and scale.
    
    Returns:
        ndarray: Transformed version of img1.
    """
    circles_1 = detect_circles(img1) #(x,y) of two circles in the image
    circles_2 = detect_circles(img2)
    scale, rotation, translation = compute_transformation(circles_1, circles_2)
    transformed_image = apply_transformation(img1, scale, rotation, translation, circles_1[0])  # using the first circle as the rotation center
    return transformed_image

def get_transformed_bw_img(bw_image, scale, rotation, translation, rotation_center):
    """
    Apply a computed transformation to a black and white image.
    
    Parameters:
        bw_image (ndarray): The black and white image to be transformed.
        scale (float): Scaling factor.
        rotation (float): Rotation angle in radians.
        translation (tuple): Translation vector (dx, dy).
        rotation_center (tuple): Center point for rotation (x, y).
    
    Returns:
        ndarray: Transformed black and white image.
    """
    transformed_bw_image = apply_transformation(bw_image, scale, rotation, translation, rotation_center)
    return transformed_bw_image



def main():
    img1 = cv2.imread('front/product_only/M0_L1.bmp')
    img2 = cv2.imread('front/product_only/M1_L0.bmp')
    transformed_image = get_transformed_img1(img1, img2)
    
    cv2.imshow("Original Image", img1)
    cv2.imshow("Original Image 2", img2)
    cv2.imshow("Transformed Image", transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
