# coding: utf-8

# Standard imports
import logging
import random
import os
from PIL import Image

# External imports
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt


def show_image(X):
    num_c = X.shape[0]
    plt.figure()
    plt.imshow(X[0] if num_c == 1 else X.permute(1, 2, 0), cmap='gray')
    plt.show()


# --- CLASSE POUR LE TEST (AJOUTÉE) ---
# Nécessaire car le dossier de test du challenge n'a pas de sous-dossiers par classe
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # On liste les images (jpg, png, etc.)
        self.image_files = [
            f for f in os.listdir(root_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        # On force le chargement en noir & blanc (L)
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_name


def get_dataloaders(data_config, use_cuda):
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]

    logging.info("  - Dataset creation")

    # Transformation adaptée pour le plancton (Gris + 128x128)
    input_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # --- MODIFICATION ICI : ImageFolder au lieu de Caltech101 ---
    # ImageFolder s'attend à une structure : trainpath/classe_A/image.jpg
    try:
        base_dataset = torchvision.datasets.ImageFolder(
            root=data_config["trainpath"],
            transform=input_transform,
        )
    except Exception as e:
        logging.error(f"Erreur chargement données : {e}")
        # Fallback pour debug local si le chemin est faux
        raise e

    logging.info(f"  - I loaded {len(base_dataset)} samples")

    indices = list(range(len(base_dataset)))
    random.shuffle(indices)
    num_valid = int(valid_ratio * len(base_dataset))
    train_indices = indices[num_valid:]
    valid_indices = indices[:num_valid]

    train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(base_dataset, valid_indices)

    # Build the dataloaders
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )

    # ImageFolder utilise 'classes' et non 'categories'
    num_classes = len(base_dataset.classes)
    input_size = tuple(base_dataset[0][0].shape)

    return train_loader, valid_loader, input_size, num_classes


# --- FONCTION POUR LE TEST (AJOUTÉE) ---
def get_test_dataloader(data_config, use_cuda):
    test_path = data_config["testpath"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]

    input_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Utilisation de notre classe custom pour lire le dossier 'vrac'
    test_dataset = TestDataset(root=test_path, transform=input_transform)
    
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    
    return torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        **kwargs
    )