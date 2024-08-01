import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage import measure

# Define U-Net Model (same as in training script)
class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )   
        
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

# Load the trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet(1).to(device)
model.load_state_dict(torch.load('unet_model.pth', map_location=device))
model.eval()

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (480, 320))
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = torch.from_numpy(img).unsqueeze(0)
    return img

# Function to run segmentation
def run_segmentation(image):
    with torch.no_grad():
        pred = model(image.to(device))
        pred = torch.sigmoid(pred)
        pred = pred.cpu().numpy()
        pred = pred[0, 0, :, :]
        pred = np.where(pred >= 0.5, 1, 0)
    return pred

# Function to apply active contours (snake)
def apply_snake(image, segmentation_result):
    # Convert segmentation result to binary image
    binary = np.uint8(segmentation_result * 255)
    
    # Find contours in the binary image
    contours = measure.find_contours(binary, 0.8)
    
    # Choose the longest contour as the initial snake
    if contours:
        init = max(contours, key=len)
    else:
        # If no contour found, create a circular initial snake
        y, x = np.ogrid[0:binary.shape[0], 0:binary.shape[1]]
        r = min(binary.shape) // 4
        cy, cx = binary.shape[0] // 2, binary.shape[1] // 2
        init = np.array([(x, y) for x, y in zip(cx + r * np.cos(np.linspace(0, 2*np.pi, 100)),
                                                cy + r * np.sin(np.linspace(0, 2*np.pi, 100)))])
    
    # Apply Gaussian filter to reduce noise
    smoothed = gaussian(image, 3, preserve_range=True)
    
    # Invert the image for better edge detection
    edge_image = 255 - cv2.cvtColor(smoothed.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Apply active contour
    snake = active_contour(edge_image,
                           init,
                           alpha=0.015,  # Length shape parameter
                           beta=10,      # Smoothness shape parameter
                           gamma=0.001)  # Step size
    return snake

# Function to apply HED
def apply_hed(image):
    # Load the pre-trained HED model
    net = cv2.dnn.readNet('./BSDS500/deploy.prototxt.txt', './BSDS500/hed_pretrained_bsds.caffemodel')
    
    # Prepare the image for HED
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(480, 320),
                                 mean=(104.00698793, 116.66876762, 122.67891434),
                                 swapRB=False, crop=False)
    
    # Set the blob as input and perform forward pass
    net.setInput(blob)
    hed_output = net.forward()
    
    # Post-process the output
    hed_output = hed_output[0, 0]
    hed_output = (hed_output * 255).astype(np.uint8)
    
    return hed_output

# Function to apply watershed
def apply_watershed( image):
    img = image

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image (Binary Inverse + Otsu's method)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological opening to remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Dilate to find sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Use distance transform and threshold to find sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # Threshold to obtain sure foreground
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Subtract sure foreground from sure background to get unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    # Apply watershed algorithm
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    # Generate the binary mask for the watershed boundaries
    boundary_mask = (markers == -1)
    binary_result = np.uint8(boundary_mask) * 255

    # Create a color block image
    color_blocks = np.zeros(img.shape, dtype=np.uint8)
    for label in np.unique(markers):
        if label == -1:
            continue
        color_blocks[markers == label] = np.random.randint(0, 255, 3).tolist()

    # Mark watershed boundaries in red on the original image
    img[markers == -1] = [255, 0, 0]

    return img, binary_result, color_blocks

def run_unet(image, model, device):
    with torch.no_grad():
        pred = model(image.to(device))
        pred = torch.sigmoid(pred)
        pred = pred.cpu().numpy()
        pred = pred[0, 0, :, :]
        pred = np.where(pred >= 0.5, 255, 0).astype(np.uint8)
    return pred

def process_image(image_path):
    # Load the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(1).to(device)
    model.load_state_dict(torch.load('unet_model.pth', map_location=device))
    model.eval()

    # Preprocess and run U-Net
    input_image = preprocess_image(image_path)
    unet_result = run_unet(input_image, model, device)

    # Run HED
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (480, 320))
    hed_result = apply_hed(original_image)

    # Save results
    cv2.imwrite('static/result1.jpg', unet_result)
    cv2.imwrite('static/result2.jpg', hed_result)

    return 'result1.jpg', 'result2.jpg'


# Main execution
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        process_image(image_path)
    else:
        print("Please provide an image path as an argument.")
    # Open file dialog to choose an image
    # Tk().withdraw()
    # image_path = askopenfilename(title="Select an image for segmentation")
    
    # if image_path:
    #     # Load and preprocess the image
    #     input_image = preprocess_image(image_path)
    #     original_image = cv2.imread(image_path)
    #     original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    #     original_image = cv2.resize(original_image, (480, 320))  # Resize to match the model's input size
        
    #     # 1. U-Net prediction
    #     unet_result = run_segmentation(input_image)
        
    #     # 2. HED prediction
    #     hed_result = apply_hed(original_image)
        
    #     # 3 & 4. Watershed binary mask and color blocks
    #     watershed_binary, watershed_color, _ = apply_watershed(original_image)
        
    #     # 5. U-Net + Watershed
    #     unet_merged = original_image.copy()
    #     unet_merged[unet_result == 1] = [255, 255, 255]  # Mark U-Net prediction in white
    #     _, unet_watershed_binary, unet_watershed_blocks= apply_watershed(unet_merged)
        
        # 6. U-Net + Watershed color blocks


        # Display results
        # plt.figure(figsize=(20, 15))
        
        # plt.subplot(3, 2, 1)
        # plt.imshow(unet_result, cmap='gray')
        # plt.title("1. U-Net Prediction")
        # plt.axis('off')
        
        # plt.subplot(3, 2, 2)
        # plt.imshow(unet_merged, cmap='gray')
        # plt.title("2. unet_merged")
        # plt.axis('off')
        
        # plt.subplot(3, 2, 3)
        # plt.imshow(watershed_binary, cmap='gray')
        # plt.title("3. Watershed Binary Mask")
        # plt.axis('off')
        
        # plt.subplot(3, 2, 4)
        # plt.imshow(watershed_color)
        # plt.title("4. Watershed Color Blocks")
        # plt.axis('off')
        
        # plt.subplot(3, 2, 5)
        # plt.imshow(unet_watershed_binary)
        # plt.title("5. U-Net + Watershed")
        # plt.axis('off')
        
        # plt.subplot(3, 2, 6)
        # plt.imshow(unet_watershed_blocks)
        # plt.title("6. U-Net + Watershed Color Blocks")
        # plt.axis('off')

        # plt.tight_layout()
        # plt.show()
    # else:
    #     print("No image selected. Exiting.")