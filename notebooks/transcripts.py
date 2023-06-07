def get_phonems(phonem_transcript: str) -> list:
    """Get the file path of a phonem transcript and return a clean list with the truth phonems."""
    with open(phonem_transcript, "r") as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]

    phonems = [line[1] for line in lines if len(line) > 1]
    phonems = [phonem.replace("</s>", "") for phonem in phonems]
    phonems = [phonem for phonem in phonems if len(phonem) > 0]

    return phonems
