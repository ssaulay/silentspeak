import os


# -------- MODEL PARAMS --------

n_frames = 75
frame_h = 75 # --> 54
frame_w = 150 # --> 80

# -------- MODE --------

data_source = "local" # "local" or "gcp"
data_size = "sample_data" # "data" or "sample_data"

# -------- LOCAL DATA PATH --------

local_data_path = os.environ.get("LOCAL_DATA_PATH")

# -------- GCP & CLOUD STORAGE --------

gcp_project = os.environ.get("GCP_PROJECT")
gcp_region = os.environ.get("GCP_REGION")
bucket_name = os.environ.get("BUCKET_NAME")
#instance = os.environ.get("INSTANCE") # uncomment when vm is defined
instance_data_path = os.environ.get("INSTANCE_DATA_PATH")


# -------- VOCAB --------

vocab_type = "p" # "p" for phonemes / "l" for letters

vocab_phonemes = [
    "a", "deux", "i", "O", "E", "S", "Z", "N", "o", "u",
    "y", "e", "w", "a~", "U~", "o~", "neuf", "p", "t", "k",
    "b", "d", "g", "f", "s", "v", "z", "m", "n", "l", "R", "j", "H"
    ]

vocab_letters = [
    "'", "-", "2", "_",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "à", "â", "ç", "è", "é", "ê", "ë", "î", "ï", "ô", "ù", "û"
    ]
