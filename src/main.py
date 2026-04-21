"""
Image Segmentation from Scratch
=================================
Stage 1 : FCN8s        — Flood Area Binary Segmentation
Stage 2 : U-Net Custom — Car Multi-class Segmentation
Stage 3 : SMP U-Net    — Car Multi-class Segmentation (Library)

학습 완료 후 results/ 폴더에 9개의 그래프 이미지가 자동 저장.
  fcn_loss.png  / fcn_iou.png       / fcn_pa.png
  unet_loss.png / unet_iou.png      / unet_pa.png
  smp_loss.png  / smp_iou_dice.png  / smp_pa.png
"""

# !pip install segmentation-models-pytorch -q

import os
import numpy as np
from glob import glob
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")           # 화면 없이 파일로 저장
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
from torchvision import models, transforms
import torchvision.io

import segmentation_models_pytorch as smp

# ─────────────────────────────────────────────
# 0. 설정 (경로 · 하이퍼파라미터)
# ─────────────────────────────────────────────
# ※ 본인 환경에 맞게 수정하세요
FLOOD_IMAGE_DIR = "/kaggle/input/datasets/faizalkarim/flood-area-segmentation/Image/*"
FLOOD_MASK_DIR  = "/kaggle/input/datasets/faizalkarim/flood-area-segmentation/Mask/*"
CAR_IMAGE_DIR   = "/kaggle/input/datasets/intelecai/car-segmentation/car-segmentation/images/*"
CAR_CLASSES_TXT = "/kaggle/input/datasets/intelecai/car-segmentation/car-segmentation/classes.txt"

RESULTS_DIR = "/kaggle/working/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE  = 16
IMG_SIZE    = 224
LR          = 5e-4
FCN_EPOCHS  = 20
UNET_EPOCHS = 15
SMP_EPOCHS  = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")
if torch.cuda.is_available():
    print(f"[GPU] {torch.cuda.get_device_name(0)}")



# ─────────────────────────────────────────────
# 1. 그래프 저장 헬퍼
# ─────────────────────────────────────────────
def save_plot(train_vals, val_vals, label, filename):
    """Train / Val 곡선을 results/ 폴더에 저장."""
    plt.figure(figsize=(8, 5))
    plt.plot(train_vals, label=f"train {label}")
    plt.plot(val_vals,   label=f"val {label}")
    plt.xlabel("Epoch")
    plt.ylabel(label)
    plt.title(label)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → saved: {save_path}")


def save_dual_plot(train_a, val_a, label_a, train_b, val_b, label_b, filename):
    """IoU + Dice 같이 두 지표를 하나의 그래프에 저장 (SMP Stage용)."""
    plt.figure(figsize=(8, 5))
    plt.plot(train_a, label=f"train {label_a}")
    plt.plot(val_a,   label=f"val {label_a}")
    plt.plot(train_b, label=f"train {label_b}", linestyle="--")
    plt.plot(val_b,   label=f"val {label_b}",   linestyle="--")
    plt.xlabel("Epoch")
    plt.title(f"{label_a} & {label_b}")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → saved: {save_path}")


# ─────────────────────────────────────────────
# 2. 데이터 전처리 유틸
# ─────────────────────────────────────────────
def filter_invalid_images(image_list, mask_list, mode="binary"):
    """채널이 3이 아닌 이미지(RGBA, Grayscale 등)를 리스트에서 제거."""
    exclude = []
    for img_path in image_list:
        img = np.array(Image.open(img_path))
        if mode == "binary":
            if img.ndim != 3 or img.shape[2] != 3:
                exclude.append(img_path)
        else:
            if img.ndim != 3:
                exclude.append(img_path)

    for ex in exclude:
        if ex in image_list:
            image_list.remove(ex)
        if mode == "binary":
            mask_name = ex.replace("Image", "Mask").replace("jpg", "png")
        else:
            mask_name = ex.replace("images", "masks")
        if mask_name in mask_list:
            mask_list.remove(mask_name)

    print(f"  제거된 이미지 수: {len(exclude)}")
    return image_list, mask_list


# ─────────────────────────────────────────────
# 3. Dataset 클래스
# ─────────────────────────────────────────────
class FloodDataset(Dataset):
    """Stage 1 — Binary Segmentation용 Dataset."""
    def __init__(self, image_list, mask_list, transform):
        self.image_list = image_list
        self.mask_list  = mask_list
        self.transform  = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        mask_path  = image_path.replace("Image", "Mask").replace("jpg", "png")

        image = Image.open(image_path)
        mask  = Image.open(mask_path)

        image = self.transform(image)
        mask  = self.transform(mask)
        mask  = (mask > 0.5).float()    # Binary화: 0 또는 1

        return image, mask


class CarDataset(Dataset):
    """Stage 2, 3 — Multi-class Segmentation용 Dataset."""
    def __init__(self, image_list, mask_list, transform_img, transform_mask):
        self.image_list    = image_list
        self.mask_list     = mask_list
        self.transform_img  = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        mask_path  = image_path.replace("images", "masks")

        image = Image.open(image_path)
        mask  = torchvision.io.read_image(mask_path)

        if image.mode == "RGBA":            # RGBA → RGB 변환
            image = image.convert("RGB")

        image = self.transform_img(image)
        mask  = self.transform_mask(mask).squeeze(0).to(torch.long)

        return image, mask


# ─────────────────────────────────────────────
# 4. 모델 정의
# ─────────────────────────────────────────────
class FCN8s(nn.Module):
    """Stage 1, 2 — VGG16 Backbone 기반 FCN8s."""
    def __init__(self, n_classes):
        super(FCN8s, self).__init__()
        vgg      = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())

        self.block3 = nn.Sequential(*features[:17])   # Conv1~3  → (256, H/8,  W/8)
        self.block4 = nn.Sequential(*features[17:24]) # Conv4    → (512, H/16, W/16)
        self.block5 = nn.Sequential(*features[24:])   # Conv5    → (512, H/32, W/32)

        self.conv6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1)

        self.conv1x1_pool3  = nn.Conv2d(256,  n_classes, kernel_size=1)
        self.conv1x1_pool4  = nn.Conv2d(512,  n_classes, kernel_size=1)
        self.conv1x1_output = nn.Conv2d(4096, n_classes, kernel_size=1)

        self.upconv2 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, padding=1)
        self.upconv8 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=8, stride=8, padding=0)

    def forward(self, x):
        p3 = self.block3(x)
        p4 = self.block4(p3)
        p5 = self.block5(p4)
        p5 = F.relu(self.conv6(p5))
        p5 = F.relu(self.conv7(p5))

        score  = self.conv1x1_output(p5)
        score  = self.upconv2(score) + self.conv1x1_pool4(p4)   # ×2 + skip(pool4)
        score  = self.upconv2(score) + self.conv1x1_pool3(p3)   # ×2 + skip(pool3)
        output = self.upconv8(score)                             # ×8 → 원본 해상도
        return output


class UNet(nn.Module):
    """Stage 2 — Encoder-Decoder + Concat Skip Connection."""
    def __init__(self, in_channels=3, out_channels=5):
        super(UNet, self).__init__()

        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            )

        # Encoder
        self.enc1 = double_conv(in_channels, 64)
        self.enc2 = double_conv(64,  128)
        self.enc3 = double_conv(128, 256)
        self.enc4 = double_conv(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = double_conv(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4    = double_conv(1024, 512)   # concat → 512+512

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3    = double_conv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2    = double_conv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1    = double_conv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat((e4, self.upconv4(b)),  dim=1))
        d3 = self.dec3(torch.cat((e3, self.upconv3(d4)), dim=1))
        d2 = self.dec2(torch.cat((e2, self.upconv2(d3)), dim=1))
        d1 = self.dec1(torch.cat((e1, self.upconv1(d2)), dim=1))

        return self.final_conv(d1)


# ─────────────────────────────────────────────
# 5. 평가 지표
# ─────────────────────────────────────────────
def iou_binary(output, mask):
    """Stage 1 — Binary IoU."""
    output       = (output > 0.5).float()
    intersection = torch.sum(output * mask)
    union        = torch.sum(output) + torch.sum(mask) - intersection
    return intersection / (union + 1e-7)


def pa_binary(output, mask):
    """Stage 1 — Binary Pixel Accuracy."""
    output  = (output > 0.5).float()
    correct = torch.sum(output == mask)
    return correct / (torch.numel(mask) + 1e-7)


def iou_multiclass(output, mask, num_classes):
    """Stage 2 — Mean IoU (클래스별 계산 후 평균)."""
    output        = torch.argmax(output, dim=1)
    iou_per_class = []
    for cls in range(num_classes):
        pred_cls = (output == cls).float()
        true_cls = (mask  == cls).float()
        inter    = torch.sum(pred_cls * true_cls)
        union    = torch.sum(pred_cls) + torch.sum(true_cls) - inter
        iou_per_class.append(torch.tensor(1.0) if union == 0 else inter / (union + 1e-7))
    return torch.mean(torch.stack(iou_per_class))


def pa_multiclass(output, mask):
    """Stage 2 — Multi-class Pixel Accuracy."""
    output  = torch.argmax(output, dim=1)
    correct = torch.sum(output == mask)
    return correct.float() / torch.numel(mask)


def smp_metrics(pred, target, mode, num_classes):
    """Stage 3 — SMP 기반 IoU, Dice, Accuracy."""
    pred   = torch.argmax(pred, dim=1)
    target = target.long()
    tp, fp, fn, tn = smp.metrics.get_stats(pred, target, mode=mode, num_classes=num_classes)
    iou  = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    dice = smp.metrics.f1_score( tp, fp, fn, tn, reduction="micro")
    acc  = smp.metrics.accuracy( tp, fp, fn, tn, reduction="micro")
    return iou, dice, acc


# ─────────────────────────────────────────────
# 6. 학습 루프
# ─────────────────────────────────────────────
def train_fcn(model, train_dl, val_dl, epochs):
    """Stage 1 — FCN Binary Segmentation 학습."""
    criterion = nn.BCEWithLogitsLoss()
    optim     = torch.optim.Adam(model.parameters(), lr=LR)

    logs = {k: [] for k in ["train_loss","val_loss","train_iou","val_iou","train_pa","val_pa"]}

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        t_loss = t_iou = t_pa = 0
        for img, mask in tqdm(train_dl, desc=f"[FCN] Epoch {epoch+1}/{epochs} Train"):
            img, mask = img.to(device), mask.to(device)
            optim.zero_grad()
            out  = model(img)
            loss = criterion(out, mask)
            loss.backward()
            optim.step()
            t_loss += loss.item()
            t_iou  += iou_binary(out.detach(), mask).item()
            t_pa   += pa_binary(out.detach(), mask).item()

        n = len(train_dl)
        logs["train_loss"].append(t_loss / n)
        logs["train_iou"].append(t_iou   / n)
        logs["train_pa"].append(t_pa     / n)

        # ── Validation ──
        model.eval()
        v_loss = v_iou = v_pa = 0
        with torch.no_grad():
            for img, mask in tqdm(val_dl, desc=f"[FCN] Epoch {epoch+1}/{epochs} Val"):
                img, mask = img.to(device), mask.to(device)
                out  = model(img)
                loss = criterion(out, mask)
                v_loss += loss.item()
                v_iou  += iou_binary(out, mask).item()
                v_pa   += pa_binary(out, mask).item()

        m = len(val_dl)
        logs["val_loss"].append(v_loss / m)
        logs["val_iou"].append(v_iou   / m)
        logs["val_pa"].append(v_pa     / m)

        print(f"  [FCN] Epoch {epoch+1} | "
              f"Train Loss: {logs['train_loss'][-1]:.3f}  IoU: {logs['train_iou'][-1]:.3f}  PA: {logs['train_pa'][-1]:.3f} | "
              f"Val Loss: {logs['val_loss'][-1]:.3f}  IoU: {logs['val_iou'][-1]:.3f}  PA: {logs['val_pa'][-1]:.3f}")

    return logs


def train_unet(model, train_dl, val_dl, epochs, num_classes):
    """Stage 2 — U-Net Multi-class Segmentation 학습."""
    criterion = nn.CrossEntropyLoss()
    optim     = torch.optim.Adam(model.parameters(), lr=LR)

    logs = {k: [] for k in ["train_loss","val_loss","train_iou","val_iou","train_pa","val_pa"]}

    for epoch in range(epochs):
        model.train()
        t_loss = t_iou = t_pa = 0
        for img, mask in tqdm(train_dl, desc=f"[UNet] Epoch {epoch+1}/{epochs} Train"):
            img, mask = img.to(device), mask.to(device)
            optim.zero_grad()
            out  = model(img)
            loss = criterion(out, mask)
            loss.backward()
            optim.step()
            t_loss += loss.item()
            t_iou  += iou_multiclass(out.detach().cpu(), mask.detach().cpu(), num_classes).item()
            t_pa   += pa_multiclass(out.detach().cpu(),  mask.detach().cpu()).item()

        n = len(train_dl)
        logs["train_loss"].append(t_loss / n)
        logs["train_iou"].append(t_iou   / n)
        logs["train_pa"].append(t_pa     / n)

        model.eval()
        v_loss = v_iou = v_pa = 0
        with torch.no_grad():
            for img, mask in tqdm(val_dl, desc=f"[UNet] Epoch {epoch+1}/{epochs} Val"):
                img, mask = img.to(device), mask.to(device)
                out  = model(img)
                loss = criterion(out, mask)
                v_loss += loss.item()
                v_iou  += iou_multiclass(out.detach().cpu(), mask.detach().cpu(), num_classes).item()
                v_pa   += pa_multiclass(out.detach().cpu(),  mask.detach().cpu()).item()

        m = len(val_dl)
        logs["val_loss"].append(v_loss / m)
        logs["val_iou"].append(v_iou   / m)
        logs["val_pa"].append(v_pa     / m)

        print(f"  [UNet] Epoch {epoch+1} | "
              f"Train Loss: {logs['train_loss'][-1]:.3f}  mIoU: {logs['train_iou'][-1]:.3f}  PA: {logs['train_pa'][-1]:.3f} | "
              f"Val Loss: {logs['val_loss'][-1]:.3f}  mIoU: {logs['val_iou'][-1]:.3f}  PA: {logs['val_pa'][-1]:.3f}")

    return logs


def train_smp(model, train_dl, val_dl, epochs, num_classes):
    """Stage 3 — SMP U-Net 학습 (JaccardLoss + SMP Metrics)."""
    mode      = "multiclass"
    criterion = smp.losses.JaccardLoss(mode=mode)
    optim     = torch.optim.Adam(model.parameters(), lr=LR)

    logs = {k: [] for k in
            ["train_loss","val_loss","train_iou","val_iou",
             "train_dice","val_dice","train_pa","val_pa"]}

    for epoch in range(epochs):
        model.train()
        t_loss = t_iou = t_dice = t_pa = 0
        for img, mask in tqdm(train_dl, desc=f"[SMP] Epoch {epoch+1}/{epochs} Train"):
            img, mask = img.to(device), mask.to(device)
            optim.zero_grad()
            out        = model(img)
            loss       = criterion(out, mask)
            iou, dice, acc = smp_metrics(out.detach(), mask, mode, num_classes)
            loss.backward()
            optim.step()
            t_loss += loss.item(); t_iou += iou; t_dice += dice; t_pa += acc

        n = len(train_dl)
        logs["train_loss"].append(t_loss / n)
        logs["train_iou"].append((t_iou  / n).item())
        logs["train_dice"].append((t_dice / n).item())
        logs["train_pa"].append((t_pa   / n).item())

        model.eval()
        v_loss = v_iou = v_dice = v_pa = 0
        with torch.no_grad():
            for img, mask in tqdm(val_dl, desc=f"[SMP] Epoch {epoch+1}/{epochs} Val"):
                img, mask = img.to(device), mask.to(device)
                out        = model(img)
                loss       = criterion(out, mask)
                iou, dice, acc = smp_metrics(out, mask, mode, num_classes)
                v_loss += loss.item(); v_iou += iou; v_dice += dice; v_pa += acc

        m = len(val_dl)
        logs["val_loss"].append(v_loss / m)
        logs["val_iou"].append((v_iou  / m).item())
        logs["val_dice"].append((v_dice / m).item())
        logs["val_pa"].append((v_pa   / m).item())

        print(f"  [SMP] Epoch {epoch+1} | "
              f"Train Loss: {logs['train_loss'][-1]:.3f}  IoU: {logs['train_iou'][-1]:.3f}  Dice: {logs['train_dice'][-1]:.3f}  PA: {logs['train_pa'][-1]:.3f} | "
              f"Val Loss: {logs['val_loss'][-1]:.3f}  IoU: {logs['val_iou'][-1]:.3f}  Dice: {logs['val_dice'][-1]:.3f}  PA: {logs['val_pa'][-1]:.3f}")

    return logs


# ─────────────────────────────────────────────
# 7. 메인 실행
# ─────────────────────────────────────────────
def main():

    # ── Stage 1: FCN8s — Flood Area Binary Segmentation ──────────────────
    print("\n" + "="*60)
    print("Stage 1 | FCN8s — Flood Area Binary Segmentation")
    print("="*60)

    image_list = glob(FLOOD_IMAGE_DIR)
    mask_list  = glob(FLOOD_MASK_DIR)
    image_list, mask_list = filter_invalid_images(image_list, mask_list, mode="binary")
    print(f"  데이터: {len(image_list)}장")

    transform_flood = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ])

    flood_ds    = FloodDataset(image_list, mask_list, transform_flood)
    train_size  = int(0.8 * len(flood_ds))
    train_ds, val_ds = random_split(flood_ds, [train_size, len(flood_ds) - train_size])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    fcn_model = FCN8s(n_classes=1).to(device)
    fcn_logs  = train_fcn(fcn_model, train_dl, val_dl, FCN_EPOCHS)

    # 그래프 저장
    save_plot(fcn_logs["train_loss"], fcn_logs["val_loss"], "Loss",           "fcn_loss.png")
    save_plot(fcn_logs["train_iou"],  fcn_logs["val_iou"],  "IoU",            "fcn_iou.png")
    save_plot(fcn_logs["train_pa"],   fcn_logs["val_pa"],   "Pixel Accuracy", "fcn_pa.png")


    # ── Stage 2: U-Net — Car Multi-class Segmentation ────────────────────
    print("\n" + "="*60)
    print("Stage 2 | U-Net (Custom) — Car Multi-class Segmentation")
    print("="*60)

    image_list = glob(CAR_IMAGE_DIR)
    mask_list  = [p.replace("images", "masks") for p in image_list]
    image_list, mask_list = filter_invalid_images(image_list, mask_list, mode="multiclass")

    with open(CAR_CLASSES_TXT, "r") as f:
        cls_list    = [c.strip() for c in f.read().split(",")]
        num_classes = len(cls_list)
    print(f"  클래스: {cls_list}  ({num_classes}개)")
    print(f"  데이터: {len(image_list)}장")

    transform_img  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_mask = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE))])

    car_ds     = CarDataset(image_list, mask_list, transform_img, transform_mask)
    train_size = int(0.8 * len(car_ds))
    train_ds, val_ds = random_split(car_ds, [train_size, len(car_ds) - train_size])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    unet_model = UNet(in_channels=3, out_channels=num_classes).to(device)
    unet_logs  = train_unet(unet_model, train_dl, val_dl, UNET_EPOCHS, num_classes)

    save_plot(unet_logs["train_loss"], unet_logs["val_loss"], "Loss",           "unet_loss.png")
    save_plot(unet_logs["train_iou"],  unet_logs["val_iou"],  "mIoU",           "unet_iou.png")
    save_plot(unet_logs["train_pa"],   unet_logs["val_pa"],   "Pixel Accuracy", "unet_pa.png")


    # ── Stage 3: SMP U-Net — Car Multi-class Segmentation ────────────────
    print("\n" + "="*60)
    print("Stage 3 | SMP U-Net (ResNet34) — Car Multi-class Segmentation")
    print("="*60)

    # Stage 2와 동일한 DataLoader 재사용
    smp_model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = "imagenet",
        classes         = num_classes,
    ).to(device)

    smp_logs = train_smp(smp_model, train_dl, val_dl, SMP_EPOCHS, num_classes)

    save_plot(smp_logs["train_loss"], smp_logs["val_loss"], "Loss",           "smp_loss.png")
    save_dual_plot(
        smp_logs["train_iou"],  smp_logs["val_iou"],  "mIoU",
        smp_logs["train_dice"], smp_logs["val_dice"],  "Dice",
        "smp_iou_dice.png"
    )
    save_plot(smp_logs["train_pa"], smp_logs["val_pa"], "Pixel Accuracy", "smp_pa.png")

    print("\n✅ 모든 학습 완료! results/ 폴더에 9개의 그래프가 저장되었습니다.")


if __name__ == "__main__":
    main()
