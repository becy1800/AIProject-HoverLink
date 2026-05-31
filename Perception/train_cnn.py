import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
from torchvision import models
import torchvision.transforms as transforms

# loading dataset from csv file and images folder 
class DroneVisionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        
        # standard normalization for PyTorch models if no transform is provided
        # transform pixel colors from 0-255 to decimals 0.0-1.0 for neural network and save as tensors 
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # ToTensor() scales pixels to [0.0, 1.0]
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.labels_df.iloc[idx, 0])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # PyTorch expects RGB
        
        if self.transform:
            image = self.transform(image)

        # Get labels: [tx, ty, tc, wx, wy, wc] and retun alongside image for training 
        labels = self.labels_df.iloc[idx, 1:].values.astype('float32')
        labels = torch.tensor(labels)

        return image, labels

# CNN Architecture using ResNet18 from scratch
class TargetDetectionCNN(nn.Module):
    def __init__(self):
        super(TargetDetectionCNN, self).__init__()
        self.model = models.resnet18(weights=None)
        
        # ResNet18 expects 224x224, but works with 128x128.
        # change the final fully connected layer to output 6 values instead of 1000
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 6)

    def forward(self, x):
        return self.model(x)

# Custom composite loss for bounding box and confidence prediction 
class CompositeLoss(nn.Module):
    def __init__(self):
        super(CompositeLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none') # don't reduce yet, so we can mask

        # combines sigmoid and bce for numerical stability
        # evaluate how well the network predicts confidence (is the target visible or not?)
        self.bce = nn.BCEWithLogitsLoss() 

    def forward(self, predictions, targets):
        # predictions and targets shape: (batch_size, 6)
        # format: [tx, ty, tc, wx, wy, wc]
        
        pred_tx, pred_ty, pred_tc_logit = predictions[:, 0], predictions[:, 1], predictions[:, 2]
        pred_wx, pred_wy, pred_wc_logit = predictions[:, 3], predictions[:, 4], predictions[:, 5]
        
        targ_tx, targ_ty, targ_tc = targets[:, 0], targets[:, 1], targets[:, 2]
        targ_wx, targ_wy, targ_wc = targets[:, 3], targets[:, 4], targets[:, 5]
        
        # Confidence Loss (BCE)
        loss_tc = self.bce(pred_tc_logit, targ_tc)
        loss_wc = self.bce(pred_wc_logit, targ_wc)
        loss_conf = loss_tc + loss_wc
        
        # Coordinate Loss (MSE) - ONLY apply where target is visible (targ_conf == 1.0)
        # if the tower is not visible, target_tc will be 0, 
        # multiplying the loss by 0 effectively ignores that sample in the final loss 
        # tower coordinate loss
        mse_tx = self.mse(pred_tx, targ_tx)
        mse_ty = self.mse(pred_ty, targ_ty)
        loss_t_coord = (mse_tx + mse_ty) * targ_tc # mask out if targ_tc is 0
        loss_t_coord = loss_t_coord.mean() # average over batch
        
        # wire coordinate loss
        mse_wx = self.mse(pred_wx, targ_wx)
        mse_wy = self.mse(pred_wy, targ_wy)
        loss_w_coord = (mse_wx + mse_wy) * targ_wc # mask out if targ_wc is 0
        loss_w_coord = loss_w_coord.mean() # average over batch
        
        loss_coord = loss_t_coord + loss_w_coord
        
        # Total Loss
        total_loss = loss_coord + loss_conf
        
        return total_loss

# Training Loop Setup
def main():
    print("Setting up dataset and model...")
    dataset_csv = "clean_dataset/labels.csv"
    dataset_img = "clean_dataset/images"
    
    # ensure data exists
    if not os.path.exists(dataset_csv):
        print(f"Error: {dataset_csv} not found! Please run generate_dataset_clean.py first.")
        return

    full_dataset = DroneVisionDataset(csv_file=dataset_csv, img_dir=dataset_img)
    
    # split 80 images for training and 20 for validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # fixed seed for reproducible splits
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # feeding images in batches of 16 and shuffling for training
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # feeding images in batches of 16 without shuffling for validation
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = TargetDetectionCNN().to(device)
    criterion = CompositeLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    num_epochs = 10
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / train_size
        
        # validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
        epoch_val_loss = val_loss / val_size
        
        print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] - Train Loss: {epoch_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")

    print("Training complete!")
    torch.save(model.state_dict(), "drone_vision_model.pth")
    print("Model saved to drone_vision_model.pth")

if __name__ == "__main__":
    main()
