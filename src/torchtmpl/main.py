# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib
import json

# External imports
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo
import tqdm
import pandas as pd

# Local imports
from . import data
from . import models
from . import optim
from . import utils


def train(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])
        wandb_log = wandb.log
        wandb_log(config)
        logging.info(f"Will be recording in wandb run name : {wandb.run.name}")
    else:
        wandb_log = None

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]

    # Modification ici : on récupère aussi class_names
    train_loader, valid_loader, input_size, num_classes, class_names = data.get_dataloaders(
        data_config, use_cuda
    )

    # Build the model
    logging.info(f"= Model (Input: {input_size}, Classes: {num_classes})")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    # Build the loss
    logging.info("= Loss")
    loss = optim.get_loss(config["loss"])

    # Build the optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = optim.get_optimizer(optim_config, model.parameters())

    # Build the callbacks
    logging_config = config["logging"]
    logname = model_config["class"]
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    # Copy the config file into the logdir
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)
        
    # NOUVEAU : Sauvegarder les noms des classes pour la soumission
    with open(logdir / "classes.json", "w") as f:
        json.dump(class_names, f)

    # Make a summary script of the experiment
    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        + (f" Wandb run name : {wandb.run.name}\n\n" if wandb_log is not None else "")
        + "## Summary of the model architecture\n"
        + f"{torchinfo.summary(model, input_size=(1, *input_size))}\n\n"
        + "## Loss\n\n"
        + f"{loss}\n\n"
    )
    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)
    if wandb_log is not None:
        wandb.log({"summary": summary_text})

    # Define the early stopping callback
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=True
    )

    for e in range(config["nepochs"]):
        # Train 1 epoch
        train_loss = utils.train(model, train_loader, loss, optimizer, device)

        # Test
        test_loss = utils.test(model, valid_loader, loss, device)

        updated = model_checkpoint.update(test_loss)
        logging.info(
            "[%d/%d] Test loss : %.3f %s"
            % (
                e,
                config["nepochs"],
                test_loss,
                "[>> BETTER <<]" if updated else "",
            )
        )

        # Update the dashboard
        metrics = {"train_CE": train_loss, "test_CE": test_loss}
        if wandb_log is not None:
            logging.info("Logging on wandb")
            wandb_log(metrics)


def test(config):
    # Cette fonction est appelée pour générer le fichier de soumission
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # On récupère le dossier de logs depuis l'argument passé (le chemin du config.yaml)
    # Exemple sys.argv[1] = "logs/VanillaCNN_0/config.yaml"
    config_path = pathlib.Path(sys.argv[1])
    log_dir = config_path.parent
    
    model_path = log_dir / "best_model.pt"
    classes_path = log_dir / "classes.json"

    if not model_path.exists() or not classes_path.exists():
        logging.error("Model or classes.json not found in the log directory.")
        sys.exit(-1)

    # Chargement des classes
    with open(classes_path, "r") as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    logging.info(f"Loaded {num_classes} classes.")

    # Chargement du DataLoader de TEST (utilisation de testpath)
    data_config = config["data"]
    test_loader = data.get_test_dataloader(data_config, use_cuda)
    
    # On a besoin de la taille d'entrée pour construire le modèle
    # On prend une image du loader pour vérifier
    sample_img, _ = next(iter(test_loader))
    input_size = sample_img.shape[1:]

    # Reconstruction du modèle
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    
    # Chargement des poids
    logging.info(f"Loading weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    logging.info("Starting inference...")
    predictions = []
    filenames = []

    with torch.no_grad():
        for inputs, fnames in tqdm.tqdm(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            # On prend la classe avec la plus haute probabilité
            _, preds = torch.max(outputs, 1)
            
            predictions.extend(preds.cpu().numpy())
            filenames.extend(fnames)

    # Conversion des indices en label numérique 1..N pour matcher les dossiers 0..N-1
    predicted_labels = [str(p + 1) for p in predictions]

    # Création du DataFrame et sauvegarde CSV
    df = pd.DataFrame({
        "image_id": filenames,
        "label": predicted_labels
    })
    
    submission_file = log_dir / "submission.csv"
    df.to_csv(submission_file, index=False)
    logging.info(f"Done! Submission file saved to: {submission_file}")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]
    eval(f"{command}(config)")