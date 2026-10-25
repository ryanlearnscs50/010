"""
Step 0: download the public NHANES files used in this study (audiometry exam, audiometry questionnaire, demographics).

Run from the 5_resubmission folder:  python -I code/00_download.py
"""
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parents[1] / "data_raw"
RAW.mkdir(exist_ok=True)

FILES = {
    "1999": ["AUX1", "AUQ", "DEMO"],
    "2001": ["AUX_B", "AUQ_B", "DEMO_B"],
    "2003": ["AUX_C", "AUQ_C", "DEMO_C"],
    "2005": ["AUX_D", "AUQ_D", "DEMO_D"],
    "2009": ["AUX_F", "AUQ_F", "DEMO_F"],
    "2011": ["AUX_G", "AUQ_G", "DEMO_G"],
    "2015": ["AUX_I", "AUQ_I", "DEMO_I"],
    "2017": ["AUX_J", "AUQ_J", "DEMO_J"],
}

for year, names in FILES.items():
    for n in names:
        dest = RAW / f"{n}.xpt"
        if dest.exists():
            continue
        url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{n}.xpt"
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        if b"<html" in r.content[:500].lower():
            raise RuntimeError(f"{url} returned an HTML page instead of an XPT file")
        dest.write_bytes(r.content)
        print(f"{n}.xpt  {len(r.content):,} bytes")
print("done")
