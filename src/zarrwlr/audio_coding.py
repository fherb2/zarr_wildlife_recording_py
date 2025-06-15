

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
                            
    def __init__(self, code, sample_rate):  # <-- Das war das Problem!
        self.code = code
        self.sample_rate = sample_rate
        
    @classmethod
    def from_string_or_enum(cls, value):
        """Konvertiert String oder gibt Enum-Wert direkt zurück."""
        if isinstance(value, cls):
            return value
        elif isinstance(value, str):
            try:
                return cls[value.upper()]
            except KeyError:
                raise ValueError(f"Unknown target format: {value}")
        else:
            raise TypeError(f"Expected str or {cls.__name__}, got {type(value)}")
        
    def auto_convert_format(func):
        def wrapper(target_format, *args, **kwargs):
            # Automatische Konvertierung hier:
            converted_format = TargetFormats.from_string_or_enum(target_format)
            return func(converted_format, *args, **kwargs)
        return wrapper

    @auto_convert_format
    def process_audio(target_format):  # Bekommt immer ein Enum
        print(f"Processing: {target_format.code}")
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
    RESAMPLING_8000 = ("resampling", 8000)
    RESAMPLING_16000 = ("resampling", 16000)
    RESAMPLING_22050 = ("resampling", 22050)
    RESAMPLING_24000 = ("resampling", 24000)
    RESAMPLING_32000 = ("resampling", 32000)
    RESAMPLING_44100 = ("resampling", 44100)
    RESAMPLING_48000 = ("resampling", 48000)
    RESAMPLING_88200 = ("resampling", 88200)
    RESAMPLING_96000 = ("resampling", 96000)
    REINTERPRETING_AUTO = ("reinterpreting", None)
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
        
    @classmethod
    def from_string_or_enum(cls, value):
        """Konvertiert String oder gibt Enum-Wert direkt zurück."""
        if isinstance(value, cls):
            return value
        elif isinstance(value, str):
            try:
                return cls[value.upper()]
            except KeyError:
                raise ValueError(f"Unknown target format: {value}")
        else:
            raise TypeError(f"Expected str or {cls.__name__}, got {type(value)}")
        
    def auto_convert_format(func):
        def wrapper(target_format, *args, **kwargs):
            # Automatische Konvertierung hier:
            converted_format = TargetFormats.from_string_or_enum(target_format)
            return func(converted_format, *args, **kwargs)
        return wrapper

    @auto_convert_format
    def process_audio(target_format):  # Bekommt immer ein Enum
        print(f"Processing: {target_format.code}")
#
# End of Class TargetSamplingTransforming
#
# ############################################################
# ############################################################


# ############################################################
# ############################################################
#
# Class AudioCompressionBaseType
# ==============================
#
# Differentiates between uncompressed, lossy and lossless compressed
#
class AudioCompressionBaseType(Enum):
    UNCOMPRESSED = "uncompressed"
    LOSSLESS_COMPRESSED = "lossless_compressed"
    LOSSY_COMPRESSED = "lossy_compressed"
    UNKNOWN = "unknown"
#
# End of Class AudioCompressionBaseType
#
# ############################################################
# ############################################################


# ############################################################
# ############################################################
#
# Method audio_codec_compression()
# ================================
#
# Recognises AudioCompressionBaseType from audio codec_name
#
def get_audio_codec_compression_type(codec_name: str) -> AudioCompressionBaseType:
    """
    Erkennt den Kompressionstyp anhand des Codec-Namens.
    Unterstützt alle gängigen Audio-Codecs, die ffmpeg dekodieren kann.
    
    Args:
        codec_name: Name des Audio-Codecs
        
    Returns:
        AudioCompressionBaseType: Kompressionstyp (UNCOMPRESSED, LOSSLESS_COMPRESSED, LOSSY_COMPRESSED, UNKNOWN)
    """
    # Normalisierung des Codec-Namens (lowercase, Bindestriche entfernen)
    normalized_name = codec_name.lower().replace("-", "_")
    
    # UNCOMPRESSED - PCM und verwandte unkomprimierte Formate
    if (normalized_name.startswith("pcm_") or 
        normalized_name in {"pcm", "s16le", "s16be", "s24le", "s24be", "s32le", "s32be", 
                           "f32le", "f32be", "f64le", "f64be", "u8", "s8"}):
        return AudioCompressionBaseType.UNCOMPRESSED
    
    # LOSSLESS_COMPRESSED - Verlustfreie Kompression
    lossless_codecs = {
        # Weit verbreitete verlustfreie Codecs
        "flac", "alac", "wavpack", "ape", "tak", "tta", "wv",
        # Monkey's Audio
        "monkeys_audio",
        # OptimFROG
        "ofr", "optimfrog",
        # WavPack Varianten
        "wavpack", "wvpk",
        # ALAC Varianten
        "apple_lossless", "m4a_lossless",
        # Weitere verlustfreie
        "mlp", "truehd", "shorten", "shn", "als",
        # Real Audio Lossless
        "ralf"
    }
    
    if normalized_name in lossless_codecs:
        return AudioCompressionBaseType.LOSSLESS_COMPRESSED
    
    # LOSSY_COMPRESSED - Verlustbehaftete Kompression
    lossy_codecs = {
        # MPEG Audio Familie
        "mp3", "mp2", "mp1", "mpa", "mpega", "mp1float", "mp2float", "mp3float",
        "mp3adu", "mp3adufloat", "mp3on4", "mp3on4float",
        # AAC Familie
        "aac", "aac_low", "aac_main", "aac_ssr", "aac_ltp", "aac_he", "aac_he_v2",
        "libfdk_aac", "aac_fixed", "aac_at", "aac_latm",
        # Opus und Vorbis
        "opus", "libopus", "vorbis", "libvorbis", "ogg",
        # AC-3 Familie
        "ac3", "eac3", "ac3_fixed", "eac3_core",
        # Windows Media Audio
        "wma", "wmav1", "wmav2", "wmapro", "wmavoice",
        # Note: wmalossless ist eigentlich verlustfrei, aber ffmpeg behandelt es manchmal als lossy
        "wmalossless",
        # Dolby Codecs
        "dts", "dca", "dtshd", "dtse",
        # AMR (Adaptive Multi-Rate)
        "amrnb", "amrwb",
        # Weitere verlustbehaftete Codecs
        "adpcm_ima_wav", "adpcm_ms", "adpcm_g726", "adpcm_yamaha",
        "g722", "g723_1", "g729", "g726", "g726le", "gsm", "gsm_ms",
        "qcelp", "evrc", "sipr", "cook", "atrac1", "atrac3",
        "ra_144", "ra_288", "nellymoser", "truespeech",
        "qdm2", "imc", "mace3", "mace6",
        "adx", "xa", "sol_dpcm", "interplay_dpcm",
        "roq_dpcm", "xan_dpcm", "sdx2_dpcm",
        # Spezialisierte/seltene Codecs
        "bmv_audio", "dsicinaudio", "smackaudio", "ws_snd1",
        "paf_audio", "on2avc", "binkaudio_rdft", "binkaudio_dct",
        "qdmc", "speex", "libspeex",
        # Game Audio Codecs
        "vmdaudio", "4xm",
        # Broadcast/Professional
        "aptx", "aptx_hd", "sbc", "ldac",
        # Real Audio
        "real_144", "real_288",
        # 8SVX (Amiga)
        "8svx_exp", "8svx_fib",
        # Musepack
        "mpc7", "mpc8",
        # Weitere ADPCM Varianten (alle verlustbehaftet)
        "adpcm_4xm", "adpcm_adx", "adpcm_afc", "adpcm_agm", "adpcm_aica",
        "adpcm_argo", "adpcm_ct", "adpcm_dtk", "adpcm_ea", "adpcm_ea_maxis_xa",
        "adpcm_ea_r1", "adpcm_ea_r2", "adpcm_ea_r3", "adpcm_ea_xas",
        "adpcm_ima_alp", "adpcm_ima_amv", "adpcm_ima_apc", "adpcm_ima_apm",
        "adpcm_ima_cunning", "adpcm_ima_dat4", "adpcm_ima_dk3", "adpcm_ima_dk4",
        "adpcm_ima_ea_eacs", "adpcm_ima_ea_sead", "adpcm_ima_iss", "adpcm_ima_moflex",
        "adpcm_ima_mtf", "adpcm_ima_oki", "adpcm_ima_qt", "adpcm_ima_rad",
        "adpcm_ima_smjpeg", "adpcm_ima_ssi", "adpcm_ima_ws",
        "adpcm_mtaf", "adpcm_psx", "adpcm_sbpro_2", "adpcm_sbpro_3", "adpcm_sbpro_4",
        "adpcm_swf", "adpcm_thp", "adpcm_thp_le", "adpcm_vima", "adpcm_xa", "adpcm_zork",
        # DPCM Varianten
        "derf_dpcm", "gremlin_dpcm",
        # Weitere Sprach-Codecs
        "acelp.kelvin", "libcodec2", "libgsm", "libgsm_ms", "ilbc", "dss_sp",
        # ATRAC Varianten
        "atrac3al", "atrac3plus", "atrac3plusal", "atrac9",
        # Weitere
        "fastaudio", "hca", "hcom", "iac", "interplayacm", "metasound",
        "smackaud", "twinvq", "xma1", "xma2", "siren"
    }
    
    if normalized_name in lossy_codecs:
        return AudioCompressionBaseType.LOSSY_COMPRESSED
    
    # Spezielle Behandlung für verschiedene Codec-Kategorien
    
    # DSD (Direct Stream Digital) - Spezielles unkomprimiertes Format
    if normalized_name.startswith("dsd_"):
        return AudioCompressionBaseType.UNCOMPRESSED
    
    # Dolby E - professioneller Broadcast-Codec (verlustbehaftet)
    if normalized_name == "dolby_e":
        return AudioCompressionBaseType.LOSSY_COMPRESSED
    
    # DST (Direct Stream Transfer) - verlustfreie DSD-Kompression
    if normalized_name == "dst":
        return AudioCompressionBaseType.LOSSLESS_COMPRESSED
    
    # DV Audio - verlustbehaftet
    if normalized_name == "dvaudio":
        return AudioCompressionBaseType.LOSSY_COMPRESSED
    
    # Comfort Noise - spezieller Codec für Sprachpausen
    if normalized_name == "comfortnoise":
        return AudioCompressionBaseType.LOSSY_COMPRESSED
    
    # Wave Synthesis - generiert Audio (verlustbehaftet)
    if normalized_name == "wavesynth":
        return AudioCompressionBaseType.LOSSY_COMPRESSED
    
    # S302M - professioneller Broadcast-Standard (unkomprimiert)
    if normalized_name == "s302m":
        return AudioCompressionBaseType.UNCOMPRESSED
    
    # Sonic - experimenteller verlustfreier Codec
    if normalized_name == "sonic":
        return AudioCompressionBaseType.LOSSLESS_COMPRESSED
    
    # Spezielle Behandlung für ADPCM-Varianten (meist verlustbehaftet)
    if normalized_name.startswith("adpcm_"):
        return AudioCompressionBaseType.LOSSY_COMPRESSED
    
    # Spezielle Behandlung für G.7xx Codecs (verlustbehaftet)
    if normalized_name.startswith("g7") and any(c.isdigit() for c in normalized_name):
        return AudioCompressionBaseType.LOSSY_COMPRESSED
    
    # Unbekannter Codec
    return AudioCompressionBaseType.UNKNOWN
#
# End of Method audio_codec_compression()
#
# ############################################################
# ############################################################