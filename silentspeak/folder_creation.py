
from silentspeak.params import data_path
from silentspeak.download_data import *

import os

folders = [
    "sample_data",
    "data",
    "sample_data_EN",
    "data_EN",
]

sub_folders = ["transcripts", "videos"]

folders_to_create = []

for folder in folders:
    ftc = os.path.join(data_path, folder)
    folders_to_create.append(ftc)
    for sub_folder in sub_folders:
        ftc = os.path.join(data_path, folder, sub_folder)
        folders_to_create.append(ftc)


def create_folders():

    for folder in folders_to_create:

        if not os.path.exists(folder):
            # Créer le dossier
            os.makedirs(folder)
            print(f"Dossier '{folder}' créé avec succès.")
        else:
            print(f"Dossier '{folder}' existe déjà.")


if __name__ == '__main__':
    print(folders_to_create)
