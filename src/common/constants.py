import os
import pathlib

# 경로
ROOT = pathlib.Path(__file__).parent.parent.parent.absolute()
LOG_FILEPATH: str = os.path.join(ROOT, "logs")

for path in [LOG_FILEPATH]:
    if not os.path.exists(path):
        os.makedirs(path)
