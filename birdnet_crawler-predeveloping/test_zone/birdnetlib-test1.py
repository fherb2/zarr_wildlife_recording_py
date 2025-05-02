from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime
from pprint import pprint

file = "audio_source/karlinsound_Y2023_dayOfYear150_m05_d30_H02_M45_S00.opus"

# Load and initialize the BirdNET-Analyzer models.
analyzer = Analyzer()

recording = Recording(
    analyzer,
    file,
    lat=51.0,
    lon=14.3,
    date=datetime(year=2023, month=5, day=30),
    min_conf=0.25,
)
recording.analyze()
pprint(recording.detections)

