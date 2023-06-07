"""The functions to create clean transcripts from the original database"""

import pandas as pd
import os
from loading import get_phonems


def get_all_transcripts(path: str) -> list:
    """
    input:
    >> path: The root path to the original database
    """

    all_transcripts = []

    sessions = ["I", "II"]
    files = [f"{i+1}.txt" for i in range(238)]

    for session in sessions:
        locutors = [f"Locuteur_{session}_{i+1}" for i in range(8)]

        for locutor in locutors:

            for file in files:
                #print(f"{session}-{locutor}-{file}")

                # Trouve le bon chemin pour les labels (pas toujours écrit de façon homogène)

                session_path = os.path.join(
                        path, f"Session {session}", locutor)

                session_dirs = os.listdir(session_path)

                lab_dir = [dir for dir in session_dirs if "phone" in dir.lower()][0]

                phonem_transcript = os.path.join(
                    path, f"Session {session}", locutor, lab_dir, file)
                phonems = get_phonems(phonem_transcript)

                transcript = {
                    "session" : session,
                    "locutor" : locutor,
                    "file" : file,
                    "phonem_transcript" : phonems
                }

                all_transcripts.append(transcript)

    return all_transcripts


def get_transcripts_df(all_transcripts: list) -> pd.DataFrame:
    """
    Returns a dataframe of all files and transcripts in the database

    input:
    >> all_transcripts: a list of dictionnaries where each dictionnary holds
    the information for the session, the locutor, the file and the phoneme transcript
    """

    transcripts_df = pd.DataFrame.from_dict(all_transcripts)
    transcripts_df["phonem_str"] = transcripts_df["phonem_transcript"].map(
        lambda x : ".".join(x)
    )

    def count_occurrences(phoneme):
        count = transcripts_df["phonem_str"].value_counts()[phoneme]
        return count

    transcripts_df["occurrences"] = transcripts_df["phonem_str"].map(count_occurrences)

    return transcripts_df


def get_most_frequent_phoneme(df: pd.DataFrame, file: int) -> list:
    """
    Returns the most frequent list of phonemes for the file <file>.txt
    >> df: a dataframe of all files and transcripts in the database
    >> file: <int> between 1 and 238 (i.e. 1.txt and 238.txt)
    """
    phonemes_list = df[df["file"] == f"{file}.txt"] \
        .sort_values("occurrences", ascending = False) \
        .iloc[0]["phonem_transcript"]
    return phonemes_list


def create_txt_file(output_path: str, df: pd.DataFrame, file: int):
    """
    Create a p_<file>.txt file
    input:
    >> output_path: the path where the .txt file will be saved
    >> df: a dataframe of all files and transcripts in the database
    >> file: an <int> between 1 and 238
    """

    # File name
    if file < 10:
        file_name = f"p_00{file}.txt"
    elif file < 100:
        file_name = f"p_0{file}.txt"
    else:
        file_name = f"p_{file}.txt"

    phonemes = get_most_frequent_phoneme(df = df, file = file)

    file_path = os.path.join(output_path, file_name)
    with open(file_path, "w") as writer:
        writer.writelines("0\n")
        writer.writelines("0\n")
        writer.writelines("0 </s>\n")
        for phoneme in phonemes:
            writer.writelines(f"0 {phoneme}\n")
        writer.writelines("0 </s>\n")
        writer.writelines("0\n")


def create_txt_file_letters(output_path: str, df: pd.DataFrame, file: int):
    """
    Create a l_<file>.txt file --> a transcript file where the base units are the letters of the full sentence
    input:
    >> output_path: the path where the .txt file will be saved
    >> df: a dataframe of all transcripts with full sentences in the database
    >> file: an <int> between 1 and 238
    """

    # File name
    if file < 10:
        file_name = f"l_00{file}.txt"
    elif file < 100:
        file_name = f"l_0{file}.txt"
    else:
        file_name = f"l_{file}.txt"

    sentence = df[df["Num"] == file].iloc[0]["Ok"]
    sentence = sentence.replace(" ", "_")

    file_path = os.path.join(output_path, file_name)
    with open(file_path, "w") as writer:
        writer.writelines("0\n")
        writer.writelines("0\n")
        writer.writelines("0 </s>\n")
        for char in sentence:
            writer.writelines(f"0 {char}\n")
        writer.writelines("0 </s>\n")
        writer.writelines("0\n")
