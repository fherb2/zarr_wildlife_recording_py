import soundfile as sf
with sf.SoundFile('/workspace/test_zone/audio_source/karlinsound_Y2023_dayOfYear150_m05_d30_H02_M45_S00.opus') as opusf:
    samplerate = opusf.samplerate
    print(f"{samplerate=}")
    data = opusf.read()