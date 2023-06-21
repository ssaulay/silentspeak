
# Import des modules
from silentspeak.params import data_path
from silentspeak.download_data import *

import os

# Liste des noms de dossiers à créer
folders = [
    "sample_data",
    "data",
    "sample_data_EN",
    "data_EN",
]

# Liste des sous-dossiers à créer dans chaque dossier
sub_folders = ["transcripts", "videos"]


# Initialisation d'une liste vide pour stocker les chemins complets des dossiers à créer
folders_to_create = []

# Pour chaque dossier dans la liste 'folders'...
for folder in folders:
    # Construit le chemin complet vers le dossier et l'ajoute à la liste 'folders_to_create'
    ftc = os.path.join(data_path, folder)
    folders_to_create.append(ftc)
    # Pour chaque sous-dossier dans la liste 'sub_folders'...
    for sub_folder in sub_folders:
        # Construit le chemin complet vers le sous-dossier et l'ajoute à la liste 'folders_to_create'
        ftc = os.path.join(data_path, folder, sub_folder)
        folders_to_create.append(ftc)

# Définition d'une fonction pour créer les dossiers
def create_folders():

    for folder in folders_to_create:
        # Vérifie si le dossier existe déjà
        if not os.path.exists(folder):
            # Si le dossier n'existe pas, il est créé
            os.makedirs(folder)
            print(f"Dossier '{folder}' créé avec succès.")
        else:
            # Si le dossier existe déjà, un message est affiché
            print(f"Dossier '{folder}' existe déjà.")


if __name__ == '__main__':
    print(folders_to_create)
