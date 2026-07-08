from pathlib import Path
from urllib.request import urlretrieve
import tarfile


url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
archive = Path("data/LJSpeech-1.1.tar.bz2")
archive.parent.mkdir(exist_ok=True)
if not archive.exists():
    urlretrieve(url, archive)
with tarfile.open(archive) as tar:
    tar.extractall("data")
print("data/LJSpeech-1.1")

