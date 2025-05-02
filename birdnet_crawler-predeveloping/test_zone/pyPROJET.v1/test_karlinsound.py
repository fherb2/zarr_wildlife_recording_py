
import PROJET
import delaysPROJET

PROJET.separate("karlinsound_Y2023_dayOfYear150_m05_d30_H02_M45_S00_part.wav",
         "test",
         4,
         36,
         12,
         nIter=300,
         start=1_830_000,
         len=5*48_000,
         postfix='_non-delayed')

# delaysPROJET.separate("karlinsound_Y2023_dayOfYear150_m05_d30_H02_M45_S00_part.wav",
#          "test",
#          4,
#          21,
#          5,
#          21,
#          5,
#          maxDelaySamples=3,
#          nIter=50,
#          start=1_830_000,
#          len=5*48_000,
#          postfix='_delayed')