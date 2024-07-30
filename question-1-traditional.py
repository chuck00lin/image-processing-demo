import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def watershed():
    # Read the image
    # img = cv2.imread('./dataset/question-1/sample_source.png')

    # list of images
    # 296059.jpg, 361010.jpg, 69020.jpg, 38092.jpg,147091.jpg
    img = cv2.imread('./BSDS500/images/test/100007.jpg')

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.figure()
    # plt.title('Grayscale Image')
    # plt.imshow(gray, cmap='gray')
    # plt.show()

    # Threshold the image (Binary Inverse + Otsu's method)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # plt.figure()
    # plt.title('Binary Image')
    # plt.imshow(binary, cmap='gray')
    # plt.show()

    # Morphological opening to remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    # plt.figure()
    # plt.title('Morphological Opening')
    # plt.imshow(opening, cmap='gray')
    # plt.show()

    # Dilate to find sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # plt.figure()
    # plt.title('Sure Background')
    # plt.imshow(sure_bg, cmap='gray')
    # plt.show()

    # Use distance transform and threshold to find sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # plt.figure()
    # plt.title('Distance Transform')
    # plt.imshow(dist_transform, cmap='gray')
    # plt.show()
    
    # Threshold to obtain sure foreground
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    # plt.figure()
    # plt.title('Sure Foreground')
    # plt.imshow(sure_fg, cmap='gray')
    # plt.show()

    # Subtract sure foreground from sure background to get unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    # plt.figure()
    # plt.title('Unknown Region')
    # plt.imshow(unknown, cmap='gray')
    # plt.show()

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    # plt.figure()
    # plt.title('Markers')
    # plt.imshow(markers, cmap='jet')
    # plt.colorbar()
    # plt.show()

    # Apply watershed algorithm
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    
    # Generate the boolean mask for the watershed boundaries
    boundary_mask = (markers == -1)

    # Convert the boolean mask to an 8-bit image
    boundary_image = boundary_mask.astype(np.uint8) * 255

    # Display the boolean mask
    plt.imshow(boundary_image, cmap='gray')
    plt.title('Watershed Boundary Mask')
    plt.axis('off')  # Hide the axis
    plt.show()


    # Save and display the result
    result_path = 'static/result_q1.jpg'
    cv2.imwrite(result_path, img)

    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(cv2.cvtColor(cv2.imread('./BSDS500/images/test/100007.jpg'), cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(132), plt.imshow(markers, cmap='jet'), plt.title('Markers')
    plt.subplot(133), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Segmented Image')
    plt.show()

    return result_path

watershed()
