from __future__ import annotations

from pathlib import Path


def save_uploaded_file(uploaded_file, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    file_path = destination_dir / uploaded_file.name
    with open(file_path, "wb") as file_handle:
        file_handle.write(uploaded_file.getbuffer())
    return file_path
