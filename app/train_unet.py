
from torch.nn import Module, CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from os import path
from model.resunet import ResidualUNet
import matplotlib.pyplot as plt
import time
import warnings
import numpy as np
import cv2
import torch.nn.functional as F
import os

warnings.filterwarnings('ignore')

def dice_coef(target, truth, smooth=1.0):
    target = target.contiguous().view(-1)
    truth = truth.contiguous().view(-1)
    union = target.sum() + truth.sum()
    intersection = torch.sum(target * truth)
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice

def normalize_data(output, threshold=0.7):
    output = output.cpu().detach().numpy()
    output = np.where(output > threshold, 1.0, 0)
    output = torch.from_numpy(output).to("cuda")
    return output

def run_one_epoch(model, loader, device, criterion, optimizer):
    total_loss = 0
    model.train()
    for _, (img, mask) in tqdm(enumerate(loader), total=len(loader), desc="Train"):
        img = img.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss / len(loader)

def eval(model, loader, device):
    scores = list()
    model.eval()
    with torch.no_grad():
        for _, (img, mask) in tqdm(enumerate(loader), total=len(loader), desc="Evaluate"):
            img = img.to(device)
            mask = mask.to(device)
            output = model(img)
            output = torch.sigmoid(output)
            output = normalize_data(output)
            if output.shape != mask.shape:
              _, _, h, w = mask.shape
              output = torch.nn.functional.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)

            score = dice_coef(output, mask)

            scores.append(score)
    return torch.mean(torch.stack(scores, dim=0))

def adjust_learning_rate(LEARNING_RATE, optimizer, epoch):
    lr = LEARNING_RATE * (0.8 ** (epoch // 70))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(model, traindataset, testdataset, device, epochs, criterion, optimizer, batch_size=1, save_dir="save"):
    highest_score = 0
    os.makedirs(save_dir, exist_ok=True)

    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = model.to(device)

    for ep in range(epochs):
        timer = time.time()

        learning_rate = optimizer.param_groups[0]['lr']
        adjust_learning_rate(learning_rate, optimizer, ep)

        print(f"[ Epoch {ep + 1}/{epochs} ]")
        loss_mean = run_one_epoch(model, trainloader, device, criterion, optimizer)
        train_score = eval(model, trainloader, device)
        test_score = eval(model, testloader, device)

        if test_score > highest_score:
            highest_score = test_score
            torch.save(model.state_dict(), path.join(save_dir, "best.pt"))
            print("✅ Best model updated!")

        torch.save(model.state_dict(), path.join(save_dir, "last.pt"))

        print(f"""Best Score {highest_score}
Epoch {ep+1} - Loss: {loss_mean:.4f}
Train Dice: {train_score:.4f}
Test Dice: {test_score:.4f}
Time: {round(time.time() - timer)}s
""")

from torch.utils.data import Dataset
from torchvision import transforms

class SimpleSegDataset(Dataset):
    def __init__(self, root_dir):
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.filenames = [f for f in os.listdir(self.image_dir) if f.endswith(".png")]
        self.filenames.sort()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
      name = self.filenames[idx]
      img_path = os.path.join(self.image_dir, name)
      mask_path = os.path.join(self.mask_dir, name)

      img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
      mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

      if img is None or mask is None:
          print(f"❌ 讀取失敗，略過：{name}")
          return self.__getitem__((idx + 1) % len(self))  # 換下一張，避免 DataLoader 崩潰

      img = self.transform(img)
      mask = torch.from_numpy((mask > 127).astype("float32")).unsqueeze(0)
      return img, mask



if __name__ == '__main__':
    EPOCH = 60
    BATCHSIZE = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = "/content/drive/MyDrive/unet_output"

    train_dataset = SimpleSegDataset("/content/Localization_and_Segmentation/app/VertebraSegmentation/data/train")
    test_dataset = SimpleSegDataset("/content/Localization_and_Segmentation/app/VertebraSegmentation/data/test")

    model = ResidualUNet(n_channels=1, n_classes=1)
    criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    train(model, train_dataset, test_dataset, device, EPOCH, criterion, optimizer, batch_size=BATCHSIZE, save_dir=save_dir)

    print("✅ 訓練完成，模型已儲存至：", save_dir)
