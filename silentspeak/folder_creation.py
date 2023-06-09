
from silentspeak.params import local_data_path
from silentspeak.params import instance_data_path
from silentspeak.params import data_source

if data_source == "local":
    data_path = local_data_path
else:
    data_path = instance_data_path
    pass

import os

folders = [
    'raw_data/sample_data',
    'raw_data/data',
    'raw_data/sample_data/transcripts',
    'raw_data/data/transcripts',
    'raw_data/sample_data/videos',
    'raw_data/data/videos'
]


for folder in folders:

    if not os.path.exists(folder):
        # Créer le dossier
        os.makedirs(folder)
        print(f"Dossier '{folder}' créé avec succès.")
    else:
        print(f"Dossier '{folder}' existe déjà.")
