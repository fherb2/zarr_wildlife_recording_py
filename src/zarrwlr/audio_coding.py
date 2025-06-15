

from enum import Enum

# ############################################################
# ############################################################
#
# Class TargetFormats
# ========================
# 
# Allowed target formats inclusive checking
#
class TargetFormats(Enum):
    """Defines all formats in order to save in Zarr arrays."""
    FLAC = ('flac', None)   # Automatic sampling frequency selection,
                            # Sampling frequencies from 1Hz until 655.350 kHz
                            # compression: flac (lossless)
    FLAC_8000 = ('flac', 8000)
    FLAC_16000 = ('flac', 16000)
    FLAC_22050 = ('flac', 22050)
    FLAC_24000 = ('flac', 24000)
    FLAC_32000 = ('flac', 32000)
    FLAC_44100 = ('flac', 44100)
    FLAC_48000 = ('flac', 48000)
    FLAC_88200 = ('flac', 88200)
    FLAC_96000 = ('flac', 96000)
    FLAC_176400 = ('flac', 176400)
    FLAC_192000 = ('flac', 192000)
    
    AAC = ('aac', None) # Automatic sampling frequency selection
                        # compression: lossy (Rate requested during import)
    AAC_8000 = ('aac', 8000) # AAC with defined sampling frequencies
    AAC_11025 = ('aac', 11025)
    AAC_12000 = ('aac', 12000)
    AAC_16000 = ('aac', 16000)
    AAC_22050 = ('aac', 22050)
    AAC_24000 = ('aac', 24000)
    AAC_32000 = ('aac', 32000)
    AAC_44100 = ('aac', 44100)
    AAC_48000 = ('aac', 48000)
    AAC_64000 = ('aac', 64000)
    AAC_88200 = ('aac', 88200)
    AAC_96000 = ('aac', 96000)
    
    NUMPY = ('numpy', None) # Each sampling frequency possible 
                            # (original remain preserved)
                            # compression: lossless, but only classic entropy
                            
    def __init__(self, code, sample_rate):
        self.code = code
        self.sample_rate = sample_rate
#
# End of Class TargetFormats
#
# ############################################################
# ############################################################



# ############################################################
# ############################################################
#
# Class TargetSamplingTransforming
# ================================
#
class TargetSamplingTransforming(Enum):
    EXACTLY = ("exactly", None)
    RESAMPLING_NEAREST = ("resampling_nearest", None)
    RESAMPLING_8000 = ("resampling_nearest", 8000)
    RESAMPLING_16000 = ("resampling_nearest", 16000)
    RESAMPLING_22050 = ("resampling_nearest", 22050)
    RESAMPLING_24000 = ("resampling_nearest", 24000)
    RESAMPLING_32000 = ("resampling_nearest", 32000)
    RESAMPLING_44100 = ("resampling_nearest", 44100)
    RESAMPLING_48000 = ("resampling_nearest", 48000)
    RESAMPLING_88200 = ("resampling_nearest", 88200)
    RESAMPLING_96000 = ("resampling_nearest", 96000)
    REINTERPRETING_8000 = ("reinterpreting", 8000)
    REINTERPRETING_16000 = ("reinterpreting", 16000)
    REINTERPRETING_22050 = ("reinterpreting", 22050)
    REINTERPRETING_24000 = ("reinterpreting", 24000)
    REINTERPRETING_32000 = ("reinterpreting", 32000)
    REINTERPRETING_44100 = ("reinterpreting", 44100)
    REINTERPRETING_48000 = ("reinterpreting", 48000)
    REINTERPRETING_88200 = ("reinterpreting", 88200)
    REINTERPRETING_96000 = ("reinterpreting", 96000)
    
    def __init__(self, type, sample_rate):
        self.code = type
        self.sample_rate = sample_rate
#
# End of Class TargetSamplingTransforming
#
# ############################################################
# ############################################################