import cv2
import numpy as np
from plantcv import plantcv as pcv
def process_image_q2(image_path):
    # Read the image
    img = cv2.imread(image_path, 0)
    
    # Threshold the image
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Perform connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    
    # Create a colored image to visualize the components
    colored = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(1, num_labels):  # Start from 1 to ignore background
        colored[labels == i] = np.random.randint(0, 255, size=3)
    
    # Save the result image
    result_path = 'static/result_q2.jpg'
    cv2.imwrite(result_path, colored)
    
    # Prepare analysis text
    analysis = f"Number of connected components: {num_labels - 1}\n"  # Subtract 1 to exclude background
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        analysis += f"Component {i}: Area = {area} pixels\n"
    
    return 'result_q2.jpg', analysis