import cv2
import numpy as np
from transformation import get_transformed_img1, compute_transformation, detect_circles, get_transformed_bw_img
from pre_process import pre_process


def display_defects(img_test, img_model, img_test_original, overfill_threshold=100, underfill_threshold=1000):
    """
    Detect and display defects in the test image compared to the model image.

    Parameters:
    - img_test: Binary test image.
    - img_model: Binary model image.
    - img_test_original: Original RGB image for overlay.
    - overfill_threshold: Threshold for overfill defect classification.
    - underfill_threshold: Threshold for underfill defect classification.

    Returns:
    - overlay_image: Image with defects highlighted.
    - defect: String describing detected defect type.
    """
    # Ensure images are the same size
    if img_test.shape != img_model.shape:
        raise ValueError("Both images must have the same shape.")

    # Handle case where values are 0 and 255
    if img_test.max() > 1 or img_model.max() > 1:
        img_test = img_test / 255
        img_model = img_model / 255

    # Detect underfill and overfill
    underfill_mask = (img_model == 1) & (img_test == 0)
    overfill_mask = (img_model == 0) & (img_test == 1)

    # Create an output image
    output_image = np.zeros((img_test.shape[0], img_test.shape[1], 3), dtype=np.uint8)
    # Mark underfill defects with red color
    output_image[underfill_mask] = [0, 0, 255]
    # Mark overfill defects with green color
    output_image[overfill_mask] = [0, 255, 0]

    # Apply morphological opening operation
    output_image = cv2.morphologyEx(output_image, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))

    # Recalculate after morphological operation
    underfilled = (np.sum(output_image[:,:,2] == 255) > underfill_threshold)  # Red channel
    overfilled = (np.sum(output_image[:,:,1] == 255) > overfill_threshold)   # Green channel

    if overfilled and underfilled:
        defect = "Defect type: Overfilled + Underfilled"
    elif overfilled and not underfilled:
        defect = "Defect type: Overfilled"
    elif not overfilled and underfilled:
        defect = "Defect type: Underfilled"
    else:
        defect = "No Defect!"

    # Overlay the defect display on the RGB test image
    overlay_image = cv2.addWeighted(img_test_original, 0.4, output_image, 0.7, 0)
    M = np.ones(overlay_image.shape, dtype="uint8") * 200  # 200 is the value to add; adjust as needed
    brightened_img = cv2.add(overlay_image, M)

    return overlay_image, defect


def main():
    e1 = cv2.getTickCount()

    img_model = cv2.imread('front/product_only/M0_L0.bmp')

    ###
    cv2.imshow("Raw Model", img_model)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ###
    img_test_folders = [
        'front/product_only/M-1_L-1.bmp','front/product_only/M-1_L0.bmp',
        'front/product_only/M0_L1.bmp','front/product_only/M1_L0.bmp'
    ]

    for i in range(len(img_test_folders)):
        img_test = cv2.imread(img_test_folders[i])
       
        img_model_transformed = get_transformed_img1(img_model, img_test)
       
        img_model_final = pre_process(img_model_transformed)
        img_test_final = pre_process(img_test)
    

        img_result, defect = display_defects(img_test_final, img_model_final,img_test)
        

        title = "Image " + str(i+1)+ " "+ defect
        dir = "results/" + title + ".jpg"
        cv2.imshow(title, img_result)
        cv2.imwrite(dir, img_result)

    

    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    print(f"Execution time: {time} seconds")
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()