# import des librairies
from google.cloud import storage
from silentspeak.params import bucket_name
import os


# on instancie
client = storage.Client()
bucket = client.bucket(bucket_name)

def list_files(bucket_name='silentspeak_raw_data',dataset="sample_data",datatype="transcripts", delimiter=None):
    """
    Récupère la liste des noms de fichiers dans un bucket spécifique avec un préfixe donné.

    Args:
        bucket_name (str): Nom du bucket. Par défaut : 'silentspeak_raw_data'.
        dataset (str): Nom du dataset. Par défaut : 'sample_data'.
        datatype (str): Type de données. Par défaut : 'transcripts'.
        delimiter (str): Délimiteur pour la recherche des fichiers. Par défaut : None.

    Returns:
        list: Liste des noms de fichiers présents dans le bucket avec le préfixe spécifié.
    """

    ret=[]
    prefix=dataset+"/"+datatype+""
    storage_client = storage.Client()
    # Obtient les objets blob correspondant au préfixe donné dans le bucket

    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)
    # Itère sur chaque objet blob et ajoute son nom à la liste ret
    for blob in blobs:
        ret.append(blob.name)
    # Retourne tous les éléments de la liste à partir de l'indice 1 (exclut le lien vers le dossier )

    return ret[1:]

def download_files(bucket_name='silentspeak_raw_data',dataset="sample_data",datatype="transcripts"):
    """
    Télécharge les fichiers à partir d'un bucket spécifique avec un dataset et un type de données donnés.

    Args:
    bucket_name (str): Nom du bucket. Par défaut : 'silentspeak_raw_data'.
    dataset (str): Nom du dataset. Par défaut : 'sample_data'.
    datatype (str): Type de données. Par défaut : 'transcripts'.
    """
    liste=list_files(bucket_name=bucket_name,dataset=dataset,datatype=datatype)
    # Télécharge les fichiers du bucket
    for blob_url in liste:
        blob = bucket.blob(blob_url)
        # Télécharge le fichier avec le même nom que celui du blob
        #blob.download_to_filename(blob_url)
        download_path = os.path.join("raw_data", blob_url)
        if not os.path.exists(download_path):
            blob.download_to_filename(download_path)
#decommenter pour afficher le verbose

#            print(f"Telecharge {download_path}")
#        else : print(f"Fichier déjà présent")


#code à executer  pour downloader les transcripts du dossier sample data

#download_files(bucket_name='silentspeak_raw_data',dataset="sample_data",datatype="transcripts")
#print("done")
