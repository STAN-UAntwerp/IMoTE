from pathlib import Path
from imote.config import DIR_DATASET_UPLOAD, NEW_CSV_OPTION, PMLB_OPTIONS


def get_target_col(csv_path: Path) -> str:
    sidecar = csv_path.with_suffix(".target.txt")
    return sidecar.read_text().strip()


def get_uploaded_options() -> list[dict]:
    DIR_DATASET_UPLOAD.mkdir(parents=True, exist_ok=True)
    options = []
    for csv_path in sorted(DIR_DATASET_UPLOAD.glob("*.csv")):
        options.append({"label": f"CSV: {csv_path.stem}", "value": str(csv_path)})
    return options


def build_dropdown_options() -> list[dict]:
    return [
        *PMLB_OPTIONS,
        *get_uploaded_options(),
        {"label": "+ New dataset (CSV)", "value": NEW_CSV_OPTION},
    ]