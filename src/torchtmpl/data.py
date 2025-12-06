# Fichier: src/torchtmpl/data.py
# coding: utf-8
import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# --- Classe pour lire le dossier de test (qui n'a pas de sous-dossiers classes) ---
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # On liste toutes les images jpg/png
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        # Important : Convertir en Gris (L) pour être cohérent avec le modèle
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_name

# --- Fonction appelée par main.py pour l'entraînement ---
def get_dataloaders(data_config, use_cuda):
    train_path = data_config["trainpath"]
    batch_size = data_config["batch_size"]
    num_workers = data_config.get("num_workers", 2)
    valid_ratio = data_config.get("valid_ratio", 0.2)

    # Transformations : 
    # 1. Grayscale 1 canal (car VanillaCNN attend 1 canal d'après vos logs précédents)
    # 2. Resize 128x128 (car VanillaCNN a des couches fully connected fixes)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    try:
        # Chargement via ImageFolder (attend une structure dossier/classe/image.jpg)
        full_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    except Exception as e:
        print(f"Erreur critique lors du chargement des données : {e}")
        raise e
    
    # Split Train / Validation
    nb_total = len(full_dataset)
    nb_valid = int(valid_ratio * nb_total)
    nb_train = nb_total - nb_valid
    
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [nb_train, nb_valid])

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    # On retourne une image exemple pour que main.py puisse configurer la taille d'entrée du modèle
    sample_img, _ = train_dataset[0]
    return train_loader, valid_loader, sample_img.shape, len(full_dataset.classes)

# --- Fonction pour le test ---
def get_test_dataloader(data_config, use_cuda):
    test_path = data_config["testpath"]
    batch_size = data_config["batch_size"]
    num_workers = data_config.get("num_workers", 2)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    test_dataset = TestDataset(root=test_path, transform=transform)
    
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)