import cv2
import matplotlib.pyplot as plt
import scipy.io
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
import copy
import time
import math

# Define folder paths
folder = "./BSDS500/"

# Function to save prediction results including HED
def save_prediction_results(org_image, ground_truth, unet_pred, hed_pred, index):
    os.makedirs("./unet-hed-processed", exist_ok=True)
    
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(org_image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(ground_truth, cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(unet_pred, cmap='gray')
    plt.title("U-Net Prediction")
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(hed_pred, cmap='gray')
    plt.title("HED Prediction")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"./unet-hed-processed/comparison_{index}.png")
    plt.close()


# Function to load images
def load_images(folder):
    images = []
    files = os.listdir(folder)
    files = [i for i in files if '.jpg' in i]
    files.sort()
    for filename in files:
        img = cv2.imread(os.path.join(folder, filename))
        if img.shape == (481, 321, 3):
            img = np.transpose(img, (1, 0, 2))
        # Cropping
        img = img[:320, :480, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None:
            images.append(img)
    return images

# Function to load labels
def load_labels(folder):
    labels = []
    files = os.listdir(folder)
    files.sort()
    for filename in files:
        mat = scipy.io.loadmat(os.path.join(folder, filename))
        gts_len = len(mat['groundTruth'][0])
        gt_majority = np.zeros(mat['groundTruth'][0][0][0][0][1].shape)
        for i in range(gts_len):
            label = mat['groundTruth'][0][i][0][0][1]
            label = np.where(label == 0, -1, 1)
            gt_majority += label
        gt_majority = np.where(gt_majority >= -1, 1, 0)
        if gt_majority.shape == (481, 321):
            gt_majority = gt_majority.T
        # Cropping
        gt_majority = gt_majority[:320, :480]
        labels.append(gt_majority)
    return labels

# Function to save images
def save_image_pair(image, label, index):
    os.makedirs("saved_images", exist_ok=True)
    
    # Save source image
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"saved_images/source_image_{index}.png")
    plt.close()
    
    # Save ground truth
    plt.figure(figsize=(5, 5))
    plt.imshow(label, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"saved_images/ground_truth_{index}.png")
    plt.close()

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

# Define functions to calculate metrics
# Load Images
xtrain = load_images(os.path.join(folder, 'images/train')) # xtrain_org is a list of images
xtest = load_images(os.path.join(folder, 'images/test'))
xval = load_images(os.path.join(folder, 'images/val'))

# Load Ground Truths Using Majority Vote Notion
ytrain = load_labels(os.path.join(folder, 'ground_truth/train'))
ytest = load_labels(os.path.join(folder, 'ground_truth/test'))
yval = load_labels(os.path.join(folder, 'ground_truth/val'))

# Save example images
indices_to_save = [10]  # Add or modify indices as needed

for idx in indices_to_save:
    if idx < len(xtrain) and idx < len(ytrain):
        save_image_pair(xtrain[idx], ytrain[idx], idx)
        print(f"Saved image pair for index {idx}")
    else:
        print(f"Index {idx} is out of range")
# plt.figure(figsize=(10, 5))

##---Populate images by rotating (Image augmentation)
# def augment_image(img_list, split = 8):
#     my_list = []
#     for i in range(len(img_list)):
#         for j in range(split):
#             my_list.append(ndimage.rotate(img_list[i], angle=j*(360/split), reshape=False))
#     return my_list

# # This takes a while
# xtrain = augment_image(xtrain_org)
# xval = augment_image(xval_org)

# ytrain = augment_image(ytrain_org)
# yval = augment_image(yval_org)
# del xtrain_org, xval_org, ytrain_org, yval_org

#-- Stack images along a new axis

xtrain = np.vstack([i[None, :, :, :] for i in xtrain])
xtest = np.vstack([i[None, :, :, :] for i in xtest])
xval = np.vstack([i[None, :, :, :] for i in xval])

# plt.imshow(xtrain[0, :, :, :])
# plt.show()
# Stack labels along a new axis

ytrain = np.vstack([i[None, :, :] for i in ytrain])
ytest = np.vstack([i[None, :, :] for i in ytest])
yval = np.vstack([i[None, :, :] for i in yval])


import torch
## Create tensors from images
xtrain = torch.from_numpy(np.transpose(xtrain, [0, 3, 1, 2]))
xtest = torch.from_numpy(np.transpose(xtest, [0, 3, 1, 2]))
xval = torch.from_numpy(np.transpose(xval, [0, 3, 1, 2]))
## Create tensors from labels
ytrain = torch.from_numpy(ytrain).view(len(ytrain), 1, 320, 480)
ytest = torch.from_numpy(ytest).view(len(ytest), 1, 320, 480)
yval = torch.from_numpy(yval).view(len(yval), 1, 320, 480)
# change data types
xtrain = xtrain.type('torch.FloatTensor')
xtest = xtest.type('torch.FloatTensor')
xval = xval.type('torch.FloatTensor')

ytrain = ytrain.type('torch.FloatTensor')
ytest = ytest.type('torch.FloatTensor')
yval = yval.type('torch.FloatTensor')
# Adjust data, make it usable for dataloader
train_set = [[xtrain[i, :, :, :], ytrain[i, :, :]] for i in range(xtrain.shape[0])]
test_set = [[xtest[i, :, :, :], ytest[i, :, :]] for i in range(xtest.shape[0])]
val_set = [[xval[i, :, :, :], yval[i, :, :]] for i in range(xval.shape[0])]


image_datasets = {
    'train': train_set, 'val': val_set
}

# Define dataloaders
batch_size = 4
dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
}

dataset_sizes = {
    x: len(image_datasets[x]) for x in image_datasets.keys()
}

# prediction for 1 instance
# Load the trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet(1).to(device)
model.load_state_dict(torch.load('unet_model.pth', map_location=device))
model.eval()   # Set model to evaluate mode

test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
        
index = 11
#inputs, labels = next(iter(test_loader))
# Extract the specific sample and ground truth label using the index
input_image, ground_truth = test_set[index]

# Move the inputs and labels to the specified device
inputs = input_image.unsqueeze(0).to(device)
labels = ground_truth.unsqueeze(0).to(device)


pred = model(inputs)
pred = torch.sigmoid(pred)
pred = pred.data.cpu().numpy()
pred = pred[0, 0, :, :]
pred_prob = pred.copy()
pred = np.where(pred>=0.5, 1, 0)


gt = ytest.numpy()[index, 0, :, :]
org = np.transpose(xtest.numpy()[index, :, :, :], [1, 2, 0]).astype(int)
# accuracy
accuracy = np.sum(pred == gt) / (gt.shape[0] * gt.shape[1])

# precision
tp = np.sum(pred[np.where(pred == 1)] == gt[np.where(pred == 1)])
fp = np.sum(pred[np.where(pred == 1)] != gt[np.where(pred == 1)])
precision = tp / (tp + fp)

# recall
fn = np.sum(pred[np.where(pred == 0)] != gt[np.where(pred == 0)])
recall = tp / (tp + fn)

# f-value
fvalue = 2 * precision * recall / (precision + recall)

# mean Average Precision
m_ap = average_precision_score(y_true = gt.reshape(-1), y_score = pred_prob.reshape(-1))


plt.subplots(1,3,figsize=(20,20))

plt.subplot(1, 3, 1)
plt.imshow(org)
plt.title('Org')

plt.subplot(1, 3, 2)
plt.imshow(gt)
plt.title('GT')

plt.subplot(1, 3, 3)
plt.imshow(pred)
plt.title('Pred')
plt.show()

# ##--- Evaluate the model
# num_it = len(test_set)
# test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
# iterator = iter(test_loader)
# org_list = []
# gt_list = []
# pred_list = []
# accuracy_list = []
# precision_list = []
# recall_list = []
# fvalue_list = []
# map_list = []

# for i in range(num_it):
#     inputs, labels = next(iterator)
#     inputs = inputs.to(device)
#     labels = labels.to(device)

#     pred = model(inputs)
#     pred = torch.sigmoid(pred)
#     pred = pred.data.cpu().numpy()
#     pred = pred[0, 0, :, :]
    
#     # arrays
#     pred_prob = pred.copy()
#     pred = np.where(pred>=0.5, 1, 0)
#     gt = ytest.numpy()[i, 0, :, :]
#     org = np.transpose(xtest.numpy()[i, :, :, :], [1, 2, 0]).astype(int)
    
#     # metrics
#     # accuracy
#     accuracy = np.sum(pred == gt) / (gt.shape[0] * gt.shape[1])

#     # precision
#     tp = np.sum(pred[np.where(pred == 1)] == gt[np.where(pred == 1)])
#     fp = np.sum(pred[np.where(pred == 1)] != gt[np.where(pred == 1)])
#     precision = tp / (tp + fp)

#     # recall
#     fn = np.sum(pred[np.where(pred == 0)] != gt[np.where(pred == 0)])
#     recall = tp / (tp + fn)

#     # f-value
#     if (precision == 0) & (recall == 0):
#         fvalue = 0
#     else:
#         fvalue = 2 * precision * recall / (precision + recall)
        
#     # mean Average Precision
#     m_ap = average_precision_score(y_true = gt.reshape(-1), y_score = pred_prob.reshape(-1))
    
#     print(f'Accuracy: {round(accuracy, 4)}, Precision: {round(precision, 4)}, Recall: {round(recall, 4)}, F-value: {round(fvalue, 4)}, m-AP: {round(m_ap, 4)}')
    
#     org_list.append(org)
#     gt_list.append(gt)
#     pred_list.append(pred)
    
#     accuracy_list.append(accuracy)
#     precision_list.append(precision)
#     recall_list.append(recall)
#     fvalue_list.append(fvalue)
#     map_list.append(m_ap)

# # Loop through the test set and save results
# for i in range(num_it):
#     org_image = org_list[i]
#     ground_truth = gt_list[i]
#     prediction = pred_list[i]
    
#     save_prediction_results(org_image, ground_truth, prediction, i)
#     print(f"Saved prediction and ground truth for index {i}")
# Evaluation function
def evaluate_models(model, test_set, device):
    num_it = len(test_set)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    
    unet_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'fvalue': [], 'map': []}
    hed_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'fvalue': [], 'map': []}
    
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # U-Net prediction
        with torch.no_grad():
            unet_pred = model(inputs)
            unet_pred = torch.sigmoid(unet_pred)
            unet_pred = unet_pred.cpu().numpy()[0, 0]
        
        unet_pred_binary = (unet_pred >= 0.5).astype(np.uint8)
        
        # HED prediction
        org_image = np.transpose(inputs.cpu().numpy()[0], (1, 2, 0)).astype(np.uint8)
        hed_pred = apply_hed(org_image)
        hed_pred_binary = (hed_pred >= 128).astype(np.uint8)
        
        gt = labels.cpu().numpy()[0, 0]
        
        # Calculate metrics for U-Net
        unet_metrics['accuracy'].append(np.mean(unet_pred_binary == gt))
        unet_metrics['precision'].append(np.sum((unet_pred_binary == 1) & (gt == 1)) / np.sum(unet_pred_binary == 1))
        unet_metrics['recall'].append(np.sum((unet_pred_binary == 1) & (gt == 1)) / np.sum(gt == 1))
        unet_metrics['fvalue'].append(2 * unet_metrics['precision'][-1] * unet_metrics['recall'][-1] / 
                                      (unet_metrics['precision'][-1] + unet_metrics['recall'][-1]))
        unet_metrics['map'].append(average_precision_score(gt.reshape(-1), unet_pred.reshape(-1)))
        
        # Calculate metrics for HED
        hed_metrics['accuracy'].append(np.mean(hed_pred_binary == gt))
        hed_metrics['precision'].append(np.sum((hed_pred_binary == 1) & (gt == 1)) / np.sum(hed_pred_binary == 1))
        hed_metrics['recall'].append(np.sum((hed_pred_binary == 1) & (gt == 1)) / np.sum(gt == 1))
        hed_metrics['fvalue'].append(2 * hed_metrics['precision'][-1] * hed_metrics['recall'][-1] / 
                                     (hed_metrics['precision'][-1] + hed_metrics['recall'][-1]))
        hed_metrics['map'].append(average_precision_score(gt.reshape(-1), hed_pred.reshape(-1) / 255))
        
        # Save prediction results
        save_prediction_results(org_image, gt, unet_pred_binary, hed_pred, i)
        
        print(f"Processed image {i+1}/{num_it}")
    
    return unet_metrics, hed_metrics

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, roc_curve, auc

# 1. Plot metrics over the test set
# plt.figure(figsize=(10, 20))
# plt.plot(accuracy_list, label='Accuracy')
# plt.plot(precision_list, label='Precision')
# plt.plot(recall_list, label='Recall')
# plt.plot(fvalue_list, label='F1-Score')
# plt.plot(map_list, label='mAP')
# plt.xlabel('Sample Index')
# plt.ylabel('Metric Value')
# plt.title('Performance Metrics')
# plt.legend()

# # 2. Confusion Matrix
# # Flatten the list of arrays for true labels and predicted labels
# true_labels = np.concatenate(gt_list).ravel()
# predicted_labels = np.concatenate(pred_list).ravel()
# cm = confusion_matrix(true_labels, predicted_labels)
# # Normalize the confusion matrix by row (true labels)
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# # Display the confusion matrix
# disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=['Background', 'Foreground'])
# disp.plot(cmap=plt.cm.Blues)
# plt.title('Pixel-Level Confusion Matrix')
# plt.show()

# # 3. Precision-Recall Curve
# precision, recall, _ = precision_recall_curve(true_labels, np.concatenate(pred_list).ravel())
# plt.plot(recall, precision, marker='.')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')

# # 4. ROC Curve and AUC
# fpr, tpr, _ = roc_curve(true_labels, np.concatenate(pred_list).ravel())
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC)')
# plt.legend(loc="lower right")

# plt.tight_layout()
# plt.show()
if __name__ == "__main__":
    # Load the trained U-Net model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(1).to(device)
    model.load_state_dict(torch.load('unet_model.pth', map_location=device))
    model.eval()

    # Evaluate models
    unet_metrics, hed_metrics = evaluate_models(model, test_set, device)

    # Plot comparison of metrics
    metrics = ['accuracy', 'precision', 'recall', 'fvalue', 'map']
    
    plt.figure(figsize=(20, 15))
    for i, metric in enumerate(metrics):
        plt.subplot(3, 2, i+1)
        plt.plot(unet_metrics[metric], label='U-Net')
        plt.plot(hed_metrics[metric], label='HED')
        plt.title(f'{metric.capitalize()} Comparison')
        plt.xlabel('Sample Index')
        plt.ylabel(metric.capitalize())
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('./unet-hed-processed/metrics_comparison.png')
    plt.close()

    # Print average metrics
    print("Average Metrics:")
    print("U-Net:")
    for metric in metrics:
        print(f"{metric.capitalize()}: {np.mean(unet_metrics[metric]):.4f}")
    print("\nHED:")
    for metric in metrics:
        print(f"{metric.capitalize()}: {np.mean(hed_metrics[metric]):.4f}")