import cv2
import numpy as np
from plantcv import plantcv as pcv

def process_image_q2(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Convert RGB to HSV and extract the saturation channel
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')

    # Threshold the image
    s_thresh = pcv.threshold.binary(gray_img=s, threshold=36, object_type='light')

    # Analyze the object
    analysis_image = pcv.analyze.size(img=img, labeled_mask=s_thresh)

        # Save the result image
    result_path = './result_q2_test.jpg'
    cv2.imwrite(result_path, analysis_image)
    return result_path

process_image_q2('./dataset/question-2/sample_source.png')
