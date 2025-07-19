import numpy as np
import os
import torch
import torch.nn as nn
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
import piq
import torch.nn.functional as F

class DenoisingDataset(Dataset):
    def __init__(self, cleandir, corrupteddir, file_list):
        self.cleandir = cleandir
        self.corrupteddir = corrupteddir
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, id):
        filename = self.file_list[id]
        # Load image paths
        clean_path = os.path.join(self.cleandir, filename)
        corrupted_path = os.path.join(self.corrupteddir, filename)

        # Load images
        clean_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
        corrupted_img = cv2.imread(corrupted_path, cv2.IMREAD_GRAYSCALE)
        
        # Error handling    
        if clean_img is None:
            raise FileNotFoundError(f"Could not load clean image: {clean_path}")
        if corrupted_img is None:
            raise FileNotFoundError(f"Could not load corrupted image: {corrupted_path}")
        
        # Convert to float32 and normalize to [0, 1]
        clean_img = clean_img.astype(np.float32) / 255.0 # cv2.imread returns values as uint8 (0 to 255)
        corrupted_img = corrupted_img.astype(np.float32) / 255.0
        
        # Convert to torch tensors
        clean_torch = torch.tensor(clean_img).unsqueeze(0)
        corrupted_torch = torch.tensor(corrupted_img).unsqueeze(0)

        return corrupted_torch, clean_torch, filename

class MS_SSIM_Loss(nn.Module):
    def __init__(self, alpha=0.3, data_range=1.0):
        super(MS_SSIM_Loss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.ssim = piq.SSIMLoss(data_range=data_range)

    def forward(self, prediction, target):
        # Normalizes the prediction and target
        prediction = prediction.clamp(0, 1)
        target = target.clamp(0, 1)

        # obtains the loss from the functions and combines them according to the selcted weighting (alpha)
        mse_loss = self.mse(prediction, target)
        ssim_loss = self.ssim(prediction, target)
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        # Applies two 3x3 convolutional layers. Two smaller layers use less parameters and can give more non linearity
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Appending the process of the UNet architecture: encoders (downsampling) and then decoders (upsampling)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)) # needs to be transposed before upsampled
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2) # The lowest point of the U
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1) # Final steps reduces the number of channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Aggregates features (pools them)

    def forward(self, x):
        skip_connections = []
        # Downsampling path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # The bottom of the U
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upsampling
        for idx in range(0, len(self.ups), 2): # Jumps by two because of the transposition layers
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            if x.shape != skip.shape: # The shapes don't match sometimes, this corrects it
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


if __name__ == '__main__':
    def load_file_list(path):
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    # These lists are obtained by running the split_dataset.py file
    train_files = load_file_list('train_files.txt')
    val_files = load_file_list('val_files.txt')

    train_dataset = DenoisingDataset('clean_reconstruction', 'corrupted_reconstruction', train_files)
    val_dataset = DenoisingDataset('clean_reconstruction', 'corrupted_reconstruction', val_files)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("USING GPU" if torch.cuda.is_available() else "USING CPU")

    model = UNet().to(device)
    loss_func = MS_SSIM_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # The learning rate can be tweaked 

    n_epochs = 60
    patience = 8 # number of iteration without improvement before the trainer terminates
    best_val_loss = float('inf')
    epochs_without_improvement = 0 
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for corrupted, clean, _ in train_loader:
            corrupted = corrupted.to(device, dtype=torch.float32)
            clean = clean.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            output = model(corrupted)
            loss = loss_func(output, clean)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for corrupted, clean, _ in val_loader:
                corrupted = corrupted.to(device, dtype=torch.float32)
                clean = clean.to(device, dtype=torch.float32)
                output = model(corrupted)
                loss = loss_func(output, clean)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss - 1e-4:  # Slight margin of error
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'models/filepath_best.pth')
            print(f"New best model saved with val loss {best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement. ({epochs_without_improvement}/{patience})")

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    torch.save(model.state_dict(), 'models/filepath.pth')
