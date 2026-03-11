"""CRS configuration — Step 1 stub (relevant fields for fuzzer)."""

from pathlib import Path
import os

WORK_DIR: Path = Path(os.environ.get("CRS_WORK_DIR", "/tmp/crs_work"))
FUZZING_ENABLED: bool = os.environ.get("CRS_FUZZING_ENABLED", "1") == "1"
FUZZING_TIMEOUT: int = int(os.environ.get("CRS_FUZZING_TIMEOUT", "120"))
