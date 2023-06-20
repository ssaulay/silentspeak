import os

# -------- MODEL PARAMS --------

n_frames = 80 # must be at least 2x the size of transcripts sequence length
n_frames_min = 0
frame_h = 54 # --> 54
frame_w = 80 # --> 80
transcript_padding = 40 # 50 for French phonemes, 72 for french letters, 21 for English
filtered = True

# -------- MODE --------

data_source = "local" # "local" or "gcp" or "google"
data_size = "sample_data_EN" # "data", "sample_data", "data_EN" or "sample_data_EN"
model_target = os.environ.get("MODEL_TARGET")

# -------- LOCAL DATA PATH --------

local_data_path = os.environ.get("LOCAL_DATA_PATH")
test_local_video = os.environ.get("TEST_LOCAL_VIDEO")

# -------- GCP & CLOUD STORAGE --------

gcp_project = os.environ.get("GCP_PROJECT")
gcp_region = os.environ.get("GCP_REGION")
bucket_name = os.environ.get("BUCKET_NAME")
#instance = os.environ.get("INSTANCE") # uncomment when vm is defined
instance_data_path = os.environ.get("INSTANCE_DATA_PATH")
google_data_path = os.environ.get("GOOGLE_DATA_PATH")

# -------- DATA PATH --------

data_path_dict = {
    "local" : local_data_path,
    "gcp" : instance_data_path,
    "google" : google_data_path
}

data_path = data_path_dict[data_source]
models_path = os.environ.get('MODEL_PATH')

# -------- VOCAB --------

vocab_type = "l" # "p" for phonemes / "l" for letters

vocab_phonemes = [
    "a", "deux", "i", "O", "E", "S", "Z", "N", "o", "u",
    "y", "e", "w", "a~", "U~", "o~", "neuf", "p", "t", "k",
    "b", "d", "g", "f", "s", "v", "z", "m", "n", "l", "R", "j", "H"
    ]

# VOCAB LETTERS WITH ACCENTS
# vocab_letters = [
#     "'", "-", "2", "_",
#     "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
#     "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
#     "à", "â", "ç", "è", "é", "ê", "ë", "î", "ï", "ô", "ù", "û"
#     ]

# VOCAB LETTERS WITHOUT ACCENTS
vocab_letters = [
    " ", "'", "-", "2", "_",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
    ]


accents_dict = {
    "à" : "a",
    "â" : "a",
    "ç" : "c",
    "è" : "e",
    "é" : "e",
    "ê" : "e",
    "ë" : "e",
    "î" : "i",
    "ï" : "i",
    "ô" : "o",
    "ù" : "u",
    "û" : "u"
}

if vocab_type == "p":
    vocab = vocab_phonemes
else:
    vocab = vocab_letters
