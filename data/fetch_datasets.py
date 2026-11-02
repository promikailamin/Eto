#!/usr/bin/env python3
"""Download public conversation/story datasets (best effort) and convert them.

Run this where the internet is open (for example the GitHub Actions runner). Every
download is optional: a failed source is skipped and the rest of the pipeline still
works with whatever is already in data/external/.
"""
import io
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "external" / "raw"
CB = "https://raw.githubusercontent.com/gunthercox/chatterbot-corpus/master/chatterbot_corpus/data/english/"
CHATTERBOT = ("ai botprofile computers conversations emotion food gossip greetings health history "
              "humor literature money movies politics psychology science sports trivia").split()
FILES = {
    "alpaca_cleaned.json": "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json",
    "TinyStories-valid.txt": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
}
for name in CHATTERBOT:
    FILES[f"chatterbot_{name}.yml"] = f"{CB}{name}.yml"
DAILYDIALOG_ZIP = "http://yanran.li/files/ijcnlp_dailydialog.zip"


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "eto-trainer"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = RAW / name
        if target.is_file() and target.stat().st_size > 0:
            continue
        try:
            target.write_bytes(download(url))
            print(f"downloaded {name}")
        except Exception as error:  # optional source
            print(f"skipped {name}: {error}")
    target = RAW / "dialogues_text.txt"
    if not target.is_file():
        try:
            outer = zipfile.ZipFile(io.BytesIO(download(DAILYDIALOG_ZIP)))
            inner = zipfile.ZipFile(io.BytesIO(outer.read("ijcnlp_dailydialog/train/train.zip")))
            target.write_bytes(inner.read("train/dialogues_train.txt"))
            print("downloaded DailyDialog")
        except Exception as error:  # optional source
            print(f"skipped DailyDialog: {error}")
    subprocess.run([sys.executable, str(HERE / "convert_external.py")], check=True)


if __name__ == "__main__":
    main()
