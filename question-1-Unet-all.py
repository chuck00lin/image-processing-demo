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

# Define folder paths
folder = "./BSDS500/"

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


# --- Unet---Training
# Define U-net Model

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
                
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
        
        #out = torch.sigmoid(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(1)
model = model.to(device)
summary(model, input_size=(3, 320, 480))

# Define Loss Function
def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss

def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))    

def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)             

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            
            if phase == 'val':
                val_losses.append(epoch_loss)
            else:
                train_losses.append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, best_loss


# Set up device, model, optimizer, and scheduler
import torch.optim as optim
from torch.optim import lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
num_class = 1
model = UNet(num_class).to(device)

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
model, train_losses, val_losses, best_loss = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=40)

# plot train vs validation set error
plt.figure(figsize=(12,8))
plt.plot(range(len(train_losses)), train_losses, label = 'Training Set')
plt.plot(range(len(val_losses)), val_losses, label = 'Validation Set')
plt.xticks(range(len(val_losses)))
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss in Training & Validation Sets')
plt.show()

torch.save(model.state_dict(), 'unet_model.pth')

import math

model.eval()   # Set model to evaluate mode

test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
        
inputs, labels = next(iter(test_loader))
inputs = inputs.to(device)
labels = labels.to(device)

pred = model(inputs)
pred = torch.sigmoid(pred)
pred = pred.data.cpu().numpy()
pred = pred[0, 0, :, :]
pred_prob = pred.copy()
pred = np.where(pred>=0.5, 1, 0)

gt = ytest.numpy()[0, 0, :, :]
org = np.transpose(xtest.numpy()[0, :, :, :], [1, 2, 0]).astype(int)
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
print(f'Accuracy: {round(accuracy, 4)}, Precision: {round(precision, 4)}, Recall: {round(recall, 4)}, F-value: {round(fvalue, 4)}, m-AP: {round(m_ap, 4)}')

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

#---HED Model
class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
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

# Load the model.
prototxt = './BSDS500/deploy.prototxt.txt'
caffemodel = './BSDS500/hed_pretrained_bsds.caffemodel'
net = cv2.dnn.readNet(prototxt,caffemodel)

# Convert lists to NumPy arrays
xtest_np = np.array(xtest)
ytest_np = np.array(ytest)

hed_predictions = [];
for i in tqdm(range(len(xtest.numpy()))):
    img = np.transpose(xtest.numpy()[i, :, :, :], [1, 2, 0]).astype(np.uint8)
    inp = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(480,320),
                                mean=(104.00698793, 116.66876762, 122.67891434),
                                swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    hed_predictions.append(out)

accuracy_list = []
precision_list = []
recall_list = []
fvalue_list = []
map_list = []

for i in tqdm(range(len(xtest.numpy()))):
    pred = hed_predictions[i]
    gt = ytest.numpy()[i, 0, :, :]
    
    pred_prob = pred.copy()
    pred = np.where(pred>=0.5, 1, 0)
    
    hed_predictions[i] = pred
    
    # metrics
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
    if (precision == 0) & (recall == 0):
        fvalue = 0
    else:
        fvalue = 2 * precision * recall / (precision + recall)
    
    # mean Average Precision
    m_ap = average_precision_score(y_true = gt.reshape(-1), y_score = pred_prob.reshape(-1))
    
    print(f'Accuracy: {round(accuracy, 4)}, Precision: {round(precision, 4)}, Recall: {round(recall, 4)}, F-value: {round(fvalue, 4)}, m-AP: {round(m_ap, 4)}')

    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    fvalue_list.append(fvalue)
    map_list.append(m_ap)

# Inspect some images
index = 0
plt.subplots(1, 2, figsize=(10, 5))
plt.imshow(hed_predictions[index])
plt.title('Pred')
plt.show()
