# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib
import csv

# External imports
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo

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

    train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(
        data_config, use_cuda
    )

    # Build the model
    logging.info("= Model")
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
    # Let us use as base logname the class name of the modek
    logname = model_config["class"]
    if "SLURM_JOB_ID" in os.environ:
        logname += f"_{os.environ['SLURM_JOB_ID']}"
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    # Copy the config file into the logdir
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    # Make a summary script of the experiment
    input_size = next(iter(train_loader))[0].shape
    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        + (f" Wandb run name : {wandb.run.name}\n\n" if wandb_log is not None else "")
        + "## Summary of the model architecture\n"
        + f"{torchinfo.summary(model, input_size=input_size)}\n\n"
        + "## Loss\n\n"
        + f"{loss}\n\n"
        + "## Datasets : \n"
        + f"Train : {train_loader.dataset.dataset}\n"
        + f"Validation : {valid_loader.dataset.dataset}"
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
    
    if "testpath" in config["data"]:
        logging.info("=== Lancement automatique du test ===")
        test_config = config.copy()
        test_config["checkpoint"] = str(logdir / "best_model.pt")
        test(test_config)


def test(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # 1. Charger les données de test
    logging.info("= Building Test Loader")
    test_loader = data.get_test_dataloader(config["data"], use_cuda)

    # 2. Reconstruire le modèle
    input_size = (1, 128, 128) # Assurez-vous que cela correspond à votre preprocessing
    # Pour le challenge, il y a 86 classes. Il faut le spécifier en dur ou le passer en config
    num_classes = 86 
    model = models.build_model(config["model"], input_size, num_classes)
    
    # 3. Charger les poids
    logging.info(f"Loading checkpoint : {config['checkpoint']}")
    model.load_state_dict(torch.load(config['checkpoint'], map_location=device))
    model.to(device)
    model.eval()

    # 4. Inférence
    results = []
    logging.info("Starting inference...")
    with torch.no_grad():
        for inputs, filenames in tqdm.tqdm(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            for fname, pred in zip(filenames, preds):
                results.append((fname, pred))

    # 5. Création du CSV
    # On sauvegarde le CSV dans le même dossier que les logs du modèle
    save_dir = pathlib.Path(config['checkpoint']).parent
    csv_path = save_dir / "submission.csv"
    
    logging.info(f"Writing submission to {csv_path}")
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["imgname", "label"]) # Header du challenge
        for fname, label in results:
            writer.writerow([fname, label])
            
    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]
    eval(f"{command}(config)")