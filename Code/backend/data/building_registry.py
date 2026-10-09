from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = CODE_DIR.parent
DATASETS_DIR = PROJECT_ROOT / "Datasets"

BUILDINGS = {
    "a-block": {
        "name": "A Block",
        "dataset": DATASETS_DIR / "IITD" / "scored" / "transformer1_scored.csv",
    },
    "b-block": {
        "name": "B Block",
        "dataset": DATASETS_DIR / "IITD" / "scored" / "transformer2_scored.csv",
    },
    "c-block": {
        "name": "C Block",
        "dataset": DATASETS_DIR / "IITD" / "scored" / "transformer3_scored.csv",
    },
    "library": {
        "name": "Library",
        "dataset": DATASETS_DIR / "IITD" / "scored" / "library_final_scored.csv",
    },
    "hostels": {
        "name": "Hostels",
        "dataset": DATASETS_DIR / "IITD" / "scored" / "hostel_scored.csv",
    },
    "mess": {
        "name": "Mess",
        "dataset": DATASETS_DIR / "IITD" / "scored" / "mess_scored.csv",
    },
}


MAP_ORDER = ["a-block", "library", "c-block", "b-block", "ground", "hostels", "mess"]


def get_building(building_id: str) -> dict:
    if building_id not in BUILDINGS:
        raise KeyError(f"Unknown building: {building_id}")
    return BUILDINGS[building_id]
