# segmentation.py
import cv2
import numpy as np

def algorithm1(image_path):
    # Simple thresholding algorithm
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def algorithm2(image_path):
    # Simple k-means clustering algorithm
    img = cv2.imread(image_path)
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2

def process_image(image_path):
    result1 = algorithm1(image_path)
    result2 = algorithm2(image_path)
    
    # Save results
    cv2.imwrite('static/result1.jpg', result1)
    cv2.imwrite('static/result2.jpg', result2)
    
    return 'result1.jpg', 'result2.jpg'