import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.io
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import average_precision_score

# Define constants
FOLDER = "./BSDS500/"
IMAGE_SIZE = (320, 480)
BATCH_SIZE = 4

def load_images(folder):
    """Load and preprocess images from the given folder."""
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(folder, filename))
            if img.shape[:2] == (481, 321):
                img = np.transpose(img, (1, 0, 2))
            img = cv2.cvtColor(img[:IMAGE_SIZE[0], :IMAGE_SIZE[1]], cv2.COLOR_BGR2RGB)
            if img is not None:
                images.append(img)
    return images

def load_labels(folder):
    """Load and preprocess labels from the given folder."""
    labels = []
    for filename in sorted(os.listdir(folder)):
        mat = scipy.io.loadmat(os.path.join(folder, filename))
        gt_majority = np.zeros(mat['groundTruth'][0][0][0][0][1].shape)
        for gt in mat['groundTruth'][0]:
            label = np.where(gt[0][0][1] == 0, -1, 1)
            gt_majority += label
        gt_majority = np.where(gt_majority >= -1, 1, 0)
        if gt_majority.shape == (481, 321):
            gt_majority = gt_majority.T
        labels.append(gt_majority[:IMAGE_SIZE[0], :IMAGE_SIZE[1]])
    return labels

def save_image_pair(image, label, index):
    """Save an image and its corresponding label."""
    os.makedirs("saved_images", exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(f"saved_images/source_image_{index}.png")
    plt.close()
    
    plt.figure(figsize=(5, 5))
    plt.imshow(label, cmap='gray')
    plt.axis('off')
    plt.savefig(f"saved_images/ground_truth_{index}.png")
    plt.close()

# Load data
xtrain = load_images(os.path.join(FOLDER, 'images/train'))
xtest = load_images(os.path.join(FOLDER, 'images/test'))
xval = load_images(os.path.join(FOLDER, 'images/val'))

ytrain = load_labels(os.path.join(FOLDER, 'ground_truth/train'))
ytest = load_labels(os.path.join(FOLDER, 'ground_truth/test'))
yval = load_labels(os.path.join(FOLDER, 'ground_truth/val'))

# Save example images
for idx in [10]:  # Add more indices if needed
    if idx < len(xtrain) and idx < len(ytrain):
        save_image_pair(xtrain[idx], ytrain[idx], idx)
        print(f"Saved image pair for index {idx}")

# Convert to PyTorch tensors
def to_tensor(data, is_label=False):
    data = np.stack(data)
    if not is_label:
        data = np.transpose(data, (0, 3, 1, 2))
    tensor = torch.from_numpy(data).float()
    return tensor.unsqueeze(1) if is_label else tensor

xtrain, xtest, xval = map(to_tensor, [xtrain, xtest, xval])
ytrain, ytest, yval = map(lambda x: to_tensor(x, True), [ytrain, ytest, yval])

# Create datasets and dataloaders
def create_dataset(x, y):
    return [[x[i], y[i]] for i in range(len(x))]

train_set = create_dataset(xtrain, ytrain)
test_set = create_dataset(xtest, ytest)
val_set = create_dataset(xval, yval)

dataloaders = {
    'train': DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True),
    'val': DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
}

# HED Model setup
class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = self.ystart = 0
        self.xend = self.yend = 0

    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

cv2.dnn_registerLayer('Crop', CropLayer)

# Load the HED model
net = cv2.dnn.readNet('./BSDS500/deploy.prototxt.txt', './BSDS500/hed_pretrained_bsds.caffemodel')

# Predict and evaluate
def predict_and_evaluate(net, xtest, ytest):
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'fvalue': [], 'map': []}
    
    for i in tqdm(range(len(xtest))):
        img = np.transpose(xtest[i].numpy(), [1, 2, 0]).astype(np.uint8)
        inp = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=IMAGE_SIZE[::-1],
                                    mean=(104.00698793, 116.66876762, 122.67891434),
                                    swapRB=False, crop=False)
        net.setInput(inp)
        pred = net.forward()[0, 0]
        
        gt = ytest[i, 0].numpy()
        pred_binary = (pred >= 0.5).astype(int)
        
        accuracy = np.mean(pred_binary == gt)
        tp = np.sum((pred_binary == 1) & (gt == 1))
        fp = np.sum((pred_binary == 1) & (gt == 0))
        fn = np.sum((pred_binary == 0) & (gt == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fvalue = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        m_ap = average_precision_score(gt.reshape(-1), pred.reshape(-1))
        
        for metric, value in zip(metrics.keys(), [accuracy, precision, recall, fvalue, m_ap]):
            metrics[metric].append(value)
        
    return metrics

metrics = predict_and_evaluate(net, xtest, ytest)

# Print average metrics
for metric, values in metrics.items():
    print(f'Average {metric}: {np.mean(values):.4f}')

# Visualize a prediction
index = 10

# HED Prediction
plt.subplot(1, 3, 3)
img = xtest[index].permute(1, 2, 0).numpy()
blob = cv2.dnn.blobFromImage(img, 
                             scalefactor=1.0, 
                             size=IMAGE_SIZE[::-1],
                             mean=(104.00698793, 116.66876762, 122.67891434),
                             swapRB=False, crop=False)
net.setInput(blob)
pred = net.forward()[0, 0]

#---------------------------------------
# Define the functions for visualization

def generate_color_map(num_classes):
    """Generate a random color map."""
    np.random.seed(42)  # for reproducibility
    return np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)

def normalize_image(image):
    """Normalize image to [0, 1] range."""
    return (image - image.min()) / (image.max() - image.min())

def create_colored_segmentation(image, mask):
    # Ensure mask is binary and of type uint8
    mask = (mask > 0).astype(np.uint8) * 255
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a blank mask to draw filled contours
    filled_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    # Fill each contour with a unique index
    for i, contour in enumerate(contours, start=1):
        cv2.drawContours(filled_mask, [contour], 0, i, -1)
    
    # Generate colors for each unique region
    num_regions = filled_mask.max()
    colors = generate_color_map(num_regions + 1)  # +1 for background
    
    # Create colored overlay
    overlay = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
    for i in range(1, num_regions + 1):
        overlay[filled_mask == i] = colors[i]
    
    # Blend with original image
    alpha = 0.7  # Adjust for desired transparency
    colored_segmentation = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    return colored_segmentation, filled_mask

# Assuming 'pred' is your segmentation output and 'ytest' is your ground truth
original_img = xtest[index].permute(1, 2, 0).numpy()
original_img = normalize_image(original_img)  # Normalize to [0, 1] range
original_img = (original_img * 255).astype(np.uint8)  # Scale to 0-255 range
gt = ytest[index, 0].numpy()

# Create colored segmentation
colored_seg, filled_mask = create_colored_segmentation(original_img, gt)

# Visualization
plt.figure(figsize=(20, 10))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB) / 255)  # Normalize back to [0, 1] for display
plt.title('Original Image')

# Ground Truth Mask
plt.subplot(2, 2, 2)
plt.imshow(gt, cmap='gray')
plt.title('Original Ground Truth Mask')

# Filled and Labeled Mask
plt.subplot(2, 2, 3)
plt.imshow(filled_mask, cmap='nipy_spectral')
plt.title('Filled and Labeled Mask')

# Colored Segmentation Overlay
plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(colored_seg, cv2.COLOR_BGR2RGB) / 255)  # Normalize back to [0, 1] for display
plt.title('Colored Segmentation Overlay')

plt.tight_layout()
plt.show()
