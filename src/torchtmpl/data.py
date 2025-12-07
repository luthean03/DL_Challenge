# coding: utf-8

# Standard imports
import logging
import os
import glob
from PIL import Image

# External imports
import torch
import torch.utils.data
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


def show_image(X):
    num_c = X.shape[0]
    plt.figure()
    plt.imshow(X[0] if num_c == 1 else X.permute(1, 2, 0))
    plt.show()


class PlanktonTestDataset(torch.utils.data.Dataset):
    """
    Dataset spécifique pour le dossier de test qui ne contient pas de sous-dossiers de classes,
    mais directement les images.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Récupère tous les fichiers d'extensions supportées
        supported_exts = ["jpg", "jpeg", "png", "tif", "tiff", "bmp"]
        image_paths = []
        for ext in supported_exts:
            pattern = os.path.join(root_dir, "**", f"*.{ext}")
            image_paths.extend(glob.glob(pattern, recursive=True))
            pattern_upper = os.path.join(root_dir, "**", f"*.{ext.upper()}")
            image_paths.extend(glob.glob(pattern_upper, recursive=True))
        self.image_paths = sorted(image_paths)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"Aucun fichier image trouvé dans {root_dir}. Vérifie les extensions et le contenu du dossier.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        # Ouverture en RGB pour garantir 3 canaux
        image = Image.open(path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # On retourne le nom du fichier pour le CSV de soumission
        filename = os.path.basename(path)
        return image, filename


def get_dataloaders(data_config, use_cuda):
    train_path = data_config["trainpath"]  # Utilisation de l'argument trainpath
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]
    use_weighted_sampler = data_config.get("use_weighted_sampler", False)

    logging.info(f"  - Loading training data from {train_path}")

    # Transformation des images (resize obligatoire pour le batching)
    input_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Chargement du dataset complet (Train + Valid) via ImageFolder
    # ImageFolder suppose une structure : trainpath/classe/image.jpg
    base_dataset = datasets.ImageFolder(
        root=train_path,
        transform=input_transform,
    )

    logging.info(f"  - Found {len(base_dataset)} images.")

    # Séparation Train / Validation
    num_total = len(base_dataset)
    num_valid = int(valid_ratio * num_total)
    num_train = num_total - num_valid

    # Utilisation de random_split pour créer les sous-ensembles
    train_subset, valid_subset = torch.utils.data.random_split(
        base_dataset, [num_train, num_valid]
    )

    train_sampler = None
    if use_weighted_sampler:
        logging.info("  - Using weighted sampler for training")
        targets = torch.tensor(base_dataset.targets)
        class_counts = torch.bincount(targets)
        weights = 1.0 / class_counts[targets]
        subset_indices = torch.tensor(train_subset.indices)
        sampler_weights = weights[subset_indices]
        train_sampler = torch.utils.data.WeightedRandomSampler(
            sampler_weights, num_train, replacement=True
        )

    # Création des DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=use_cuda,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    # Récupération des infos utiles
    class_names = base_dataset.classes
    num_classes = len(class_names)
    input_size = tuple(base_dataset[0][0].shape)

    return train_loader, valid_loader, input_size, num_classes, class_names


def get_test_dataloader(data_config, use_cuda):
    test_path = data_config["testpath"] # Utilisation de l'argument testpath
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]
    
    logging.info(f"  - Loading test data from {test_path}")

    input_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # Utilisation du Dataset personnalisé pour le test (structure plate)
    test_dataset = PlanktonTestDataset(
        root_dir=test_path, 
        transform=input_transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda
    )
    
    return test_loader