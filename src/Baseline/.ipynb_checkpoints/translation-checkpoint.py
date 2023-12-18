import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from loadingpy import pybar

from Baseline.configuration import config
from easydict import EasyDict
cfg = EasyDict(config)

# Les imports ci-dessus sont nécessaires pour utiliser les fonctionnalités du code

class BlankStatement:
    def __init__(self):
        pass

    def __enter__(self):
        return None

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        return None

class BaselineTrainer:

    def __init__(self, quiet_mode: bool, save_name: str = 'Nothing') -> None:

        self.quiet_mode = quiet_mode
        self.create_dir(name=save_name)

    def create_dir(self, name='Nothing'):
        # Fonction pour créer un répertoire avec un nom donné ou une horodatage s'il n'est pas fourni
        if name == 'Nothing':
            import datetime
            now = datetime.datetime.now()
            name = now.strftime("%Y-%m-%d %H:%M:%S")

        self.dir = f"Models/{name}"
        os.makedirs(self.dir, exist_ok=True)

    def Init_weights(self, m):
        # Initialiser les poids des modules du modèle
        for name, param in m.named_parameters():
            if 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)

def train(self, model, train_data_loader, val_data_loader, device,
          nohup=False, mode="RNNEncDec", seq_len=cfg.sequence_length,
          load_model="Nothing", chosed_optimizer="adadelta", lr=0):
    """
    Fonction pour entraîner le modèle.

    Args:
        model (nn.Module): Le modèle à entraîner.
        train_data_loader (DataLoader): Le DataLoader pour les données d'entraînement.
        val_data_loader (DataLoader): Le DataLoader pour les données de validation.
        device (str): Le dispositif sur lequel exécuter l'entraînement (e.g., 'cuda' ou 'cpu').
        nohup (bool, optional): Si True, l'entraînement est exécuté en mode nohup (pas de barre de progression). Par défaut False.
        mode (str, optional): Le mode d'entraînement ('RNNEncDec' ou 'RNNSearch'). Par défaut "RNNEncDec".
        seq_len (int, optional): La longueur de séquence utilisée. Par défaut cfg.sequence_length.
        load_model (str, optional): Le chemin du modèle à charger avant l'entraînement. Par défaut "Nothing".
        chosed_optimizer (str, optional): L'optimiseur à utiliser ('adadelta' ou 'adam'). Par défaut "adadelta".
        lr (float, optional): Taux d'apprentissage. Si 0, utilise cfg.lr. Par défaut 0.
    """
    # ...
    
    # Initialisation du code de nom en fonction du mode et de la longueur de séquence
    name_code = "S" if mode == "RNNSearch" else "E"
    name_code += "50" if seq_len == 50 else "30"
    print("Let the training begin")
    
    self.device = device
    
    # Transférer le modèle sur le dispositif spécifié
    model = model.to(device)
    
    # Initialiser les poids du modèle si aucun modèle n'est chargé
    if load_model == "Nothing": 
        model.apply(self.Init_weights)
    else: 
        state_dict_complet = torch.load(load_model, map_location=torch.device(device))
        model.load_state_dict(state_dict_complet["model_state_dict"])

    len_batches = len(train_data_loader)

    # Définir le taux d'apprentissage si non spécifié
    if lr == 0: 
        lr = cfg.lr
    
    # Choisir l'optimiseur en fonction de l'option spécifiée
    if chosed_optimizer == "adadelta": 
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, rho=0.95, eps=1e-06)
    elif chosed_optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters())

    # Définir la fonction de perte (cross-entropy avec gestion du padding)
    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=0) 

    # Initialiser la barre de progression
    pbar = pybar(range(cfg.epochs * len_batches), base_str="training")

    # Initialiser les listes pour stocker les pertes d'entraînement et de validation
    train_losses = []
    val_losses = []

    final_loss = val_loss = None

    best_val_loss = float('inf')

    best_model = None

    # Initialiser une liste pour stocker les poids d'attention (si le mode est RNNSearch)
    if mode == "RNNSearch ":  
        best_attention_weights = []

    for epoch in range(cfg.epochs):

        # Initialiser la liste des poids d'attention pour chaque lot d'entraînement (si le mode est RNNSearch)
        if mode == "RNNSearch ": 
            attention_weights = []

        model.train()

        loss_for_one_epoch = []
        n_batch = -1

        for input_batch, output_batch  in train_data_loader:
            n_batch += 1

            input_batch, output_batch = input_batch.to(device), output_batch.to(device)

            # Initialiser les gradients à zéro
            optimizer.zero_grad()

            # Obtenir la sortie du modèle
            if mode == "RNNEncDec": 
                output = model(input_batch)
            else: 
                output, attention_weights = model(input_batch)

            # Ajuster la forme de la sortie pour le calcul de la perte
            output = output.reshape(-1, cfg.vocabulary_size)

            # Ajuster la forme des cibles pour le calcul de la perte
            output_batch_onehot = output_batch.view(-1)

            # Calculer la perte
            loss = criterion(output, output_batch_onehot)
            loss.backward()

            # Mettre à jour les poids du modèle
            optimizer.step()

            # Stocker la perte pour le lot actuel
            loss_for_one_epoch += [loss.item()]

            # Afficher les informations d'entraînement si en mode nohup
            if nohup and n_batch % (len_batches // 10) == 0: 
                print(f"{name_code} Epoch : {epoch}, batch : {n_batch}/{len_batches}, loss : {round(loss.item(), 4)}")

            # Afficher les informations d'entraînement si en mode interactif
            if not self.quiet_mode and not nohup: 
                pbar.set_description(description=f"{name_code} Epoch {epoch+1}, loss: {final_loss}, val: {val_loss}, Batch {n_batch+1}/{len_batches} loss : {round(loss.item(), 4)}")
                pbar.__next__()

            # Évaluer le modèle sur les données de validation tous les 1/3 de lots
            if n_batch + 10 % (len_batches // 3) == 0:
                val_loss = self.evaluate(model, val_data_loader, criterion, mode=mode)

                # Sauvegarder le meilleur modèle selon la perte de validation
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model.state_dict()
                    if mode == "RNNSearch ":  
                        best_attention_weights = attention_weights

                    torch.save({'model_state_dict': best_model,
                                'optimizer_state_dict': optimizer.state_dict()}, 
                                f"{self.dir}/best_model.pth")
                    model.train()

        # Calculer la perte moyenne pour l'époque d'entraînement
        final_loss = np.mean(loss_for_one_epoch)

        # Calculer la perte moyenne pour l'époque de validation
        val_loss = self.evaluate(model, val_data_loader, criterion, mode=mode)

        # Stocker les pertes d'entraînement et de validation
        train_losses += [final_loss]
        val_losses += [val_loss] 

        # Arrondir les pertes pour l'affichage
        final_loss = round(final_loss, 4)
        val_loss = round(val_loss, 4)

        # Sauvegarder les pertes sur disque
        np.save(f"{self.dir}/loss.npy", np.array(train_losses))
        np.save(f"{self.dir}/loss_val.npy", np.array(val_losses))

        # Sauvegarder le modèle si la perte de validation est la meilleure jusqu'à présent
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            if mode == "RNNSearch ": 
                best_attention_weights = attention_weights

            torch.save({'model_state_dict': best_model,
                        'optimizer_state_dict': optimizer.state_dict()}, 
                        f"{self.dir}/best_model.pth")
            torch.save(model, f"{self.dir}/best_model_usable.pth")

        # Sauvegarder le modèle après chaque époque
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, 
                    f"{self.dir}/last_model.pth")

        # Afficher les informations d'entraînement si en mode nohup
        if nohup: 
            print(f"Epoch {epoch + 1}/{cfg.epochs} : loss = {final_loss}, val_loss = {val_loss}")

    try:
        pbar.__next__()
    except StopIteration:
        print("The training has ended")

def evaluate(self, model, val_data_loader, criterion, mode='RNNEncDec'):
    """
    Évalue le modèle sur l'ensemble de validation.

    Paramètres :
    - model : Le modèle PyTorch à évaluer.
    - val_data_loader : DataLoader pour l'ensemble de validation.
    - criterion : La fonction de perte utilisée pour l'évaluation.
    - mode : Mode du modèle ('RNNEncDec' ou 'RNNSearch').

    Retourne :
    - mean_loss : La perte moyenne sur l'ensemble de validation.
    """

    # Mettre le modèle en mode évaluation
    model.eval()

    # Liste pour stocker les pertes de chaque lot
    loss_for_one_epoch = []

    # Désactiver le calcul du gradient pendant l'évaluation
    with torch.no_grad():

        # Itérer sur les lots de l'ensemble de validation
        for input_batch, output_batch in val_data_loader:

            # Déplacer les lots d'entrée et de sortie sur le périphérique spécifié
            input_batch, output_batch = input_batch.to(self.device), output_batch.to(self.device)

            # Obtenir les prédictions du modèle
            if mode == 'RNNEncDec':
                output = model(input_batch)
            else:
                output, attention_weights = model(input_batch)

            # Remodeler la sortie pour calculer la perte
            output = output.reshape(-1, cfg.vocabulary_size)

            # Aplatir output_batch pour calculer la perte
            output_batch_onehot = output_batch.view(-1)

            # Calculer la perte pour le lot
            loss = criterion(output, output_batch_onehot)

            # Ajouter la perte à la liste
            loss_for_one_epoch += [loss.item()]

    # Calculer la perte moyenne sur tous les lots
    mean_loss = np.mean(loss_for_one_epoch)

    return mean_loss
