"""NOAA MarineCadastre AIS 데이터 다운로드 스크립트 (2022년 1월)"""

import urllib.request
import zipfile
import os
from pathlib import Path
from datetime import date, timedelta

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2022"
START = date(2022, 1, 1)
END = date(2022, 1, 7)


def download_day(d: date):
    fname = f"AIS_{d.strftime('%Y_%m_%d')}"
    csv_path = RAW_DIR / f"{fname}.csv"

    if csv_path.exists():
        print(f"  [skip] {csv_path.name} 이미 존재")
        return

    url = f"{BASE_URL}/{fname}.zip"
    zip_path = RAW_DIR / f"{fname}.zip"

    print(f"  [download] {url}")
    urllib.request.urlretrieve(url, zip_path)

    print(f"  [unzip] {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)

    zip_path.unlink()
    print(f"  [done] {csv_path.name}")


if __name__ == "__main__":
    current = START
    total = (END - START).days + 1
    idx = 0

    while current <= END:
        idx += 1
        print(f"[{idx}/{total}] {current}")
        try:
            download_day(current)
        except Exception as e:
            print(f"  [error] {e}")
        current += timedelta(days=1)

    csv_count = len(list(RAW_DIR.glob("AIS_2022_01_*.csv")))
    print(f"\n완료: {csv_count}/{total} 파일 다운로드")
