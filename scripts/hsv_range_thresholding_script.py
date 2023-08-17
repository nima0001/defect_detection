import cv2
import numpy as np

# Load multiple images and convert them to HSV
# image_paths = [
#      'back/product only/B0a.bmp','back/product only/B0b.bmp', 
#      'back/product only/B1.bmp', 'back/product only/B1a.bmp'
# ]  

image_paths = [
    'front/product_only/M-1_L-1.bmp', 'front/product_only/M-1_L0.bmp',
    'front/product_only/M0_L1.bmp','front/product_only/M1_L0.bmp',
]

assert len(image_paths) >= 4, "Please provide at least 8 images for a 4x2 layout"

images = [cv2.imread(p) for p in image_paths]

# Find the size of the smallest image
min_height = min(img.shape[0] for img in images)
min_width = min(img.shape[1] for img in images)

# Resize all images to the size of the smallest image
resized_images = [cv2.resize(img, (min_width, min_height)) for img in images]
hsv_images = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in resized_images]

def update_mask(*_):
    h_low = cv2.getTrackbarPos('Hue (Low)', 'Control Window')
    h_high = cv2.getTrackbarPos('Hue (High)', 'Control Window')
    s_low = cv2.getTrackbarPos('Saturation (Low)', 'Control Window')
    s_high = cv2.getTrackbarPos('Saturation (High)', 'Control Window')
    v_low = cv2.getTrackbarPos('Value (Low)', 'Control Window')
    v_high = cv2.getTrackbarPos('Value (High)', 'Control Window')

    lower_blue = np.array([h_low, s_low, v_low])
    upper_blue = np.array([h_high, s_high, v_high])

    # Apply mask to each image and then apply morphological opening operation
    kernel = np.ones((2,2),np.uint8)  # 5x5 kernel for opening, you can adjust this size if needed
    masks = [cv2.morphologyEx(cv2.inRange(hsv, lower_blue, upper_blue), cv2.MORPH_OPEN, kernel) for hsv in hsv_images]

    # Concatenate images and masks in 4x2 layout
    orig_rows = [np.hstack(resized_images[i:i+2]) for i in range(0, 4, 2)]
    mask_rows = [np.hstack(masks[i:i+2]) for i in range(0, 4, 2)]
    
    combined_images = np.vstack(orig_rows)
    combined_masks = np.vstack(mask_rows)

    # Stack original images and masks vertically
    # final_display = np.vstack([combined_images, combined_masks])
    print(lower_blue, upper_blue)
    cv2.imshow('Images and Masks', combined_masks)

cv2.namedWindow('Control Window')

# Create trackbars
cv2.createTrackbar('Hue (Low)', 'Control Window', 0, 180, update_mask)
cv2.createTrackbar('Hue (High)', 'Control Window', 180, 180, update_mask)
cv2.createTrackbar('Saturation (Low)', 'Control Window', 0, 255, update_mask)
cv2.createTrackbar('Saturation (High)', 'Control Window', 255, 255, update_mask)
cv2.createTrackbar('Value (Low)', 'Control Window', 0, 255, update_mask)
cv2.createTrackbar('Value (High)', 'Control Window', 255, 255, update_mask)

# Display initial state
update_mask()

cv2.waitKey(0)
cv2.destroyAllWindows()
