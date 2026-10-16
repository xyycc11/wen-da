"""Explicit model setup. Not run automatically by chat or migration."""
import argparse
import hashlib
from pathlib import Path
import tarfile
import tempfile
import urllib.request

URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"


def prepare(directory):
    directory = Path(directory).resolve()
    if (directory / "onnx/model.onnx").is_file() and (directory / "onnx/tokenizer.json").is_file():
        return directory / "onnx"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as scratch:
        archive = Path(scratch) / "model.tar.gz"
        digest = hashlib.sha256()
        with urllib.request.urlopen(URL, timeout=60) as response, archive.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                target.write(chunk)
        if digest.hexdigest() != SHA256:
            raise ValueError("embedding_archive_checksum_mismatch")
        with tarfile.open(archive, "r:gz") as package:
            package.extractall(directory, filter="data")
    if not all((directory / "onnx" / name).is_file() for name in ("model.onnx", "tokenizer.json")):
        raise ValueError("embedding_archive_incomplete")
    return directory / "onnx"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=str(Path.home() / ".cache/chroma/onnx_models/all-MiniLM-L6-v2"))
    args = parser.parse_args()
    print(prepare(args.directory))


if __name__ == "__main__":
    main()
