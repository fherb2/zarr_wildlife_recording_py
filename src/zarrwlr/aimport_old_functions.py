# Erweitere import_original_audio_file() um Config-Integration:
def import_original_audio_file_with_config(
                    audio_file: str|pathlib.Path, 
                    zarr_original_audio_group: zarr.Group,
                    first_sample_time_stamp: datetime.datetime|None,
                    target_codec:str = None,  # None = use Config default
                    **kwargs):
    """
    Enhanced import with automatic config integration
    """
    # Use config defaults if not specified
    if target_codec is None:
        target_codec = 'aac'  # New default
    
    # Get AAC config if using AAC
    if target_codec == 'aac':
        aac_config = _get_aac_config_for_import()
        if 'aac_bitrate' not in kwargs:
            kwargs['aac_bitrate'] = aac_config['bitrate']
    
    # Call original function with enhanced parameters
    return import_original_audio_file(
        audio_file=audio_file,
        zarr_original_audio_group=zarr_original_audio_group,
        first_sample_time_stamp=first_sample_time_stamp,
        target_codec=target_codec,
        **kwargs
    )


def validate_aac_import_parameters(aac_bitrate: int, source_params: dict) -> dict:
    """
    Validate and optimize AAC import parameters
    
    Args:
        aac_bitrate: Requested bitrate
        source_params: Source audio parameters
        
    Returns:
        Validated and optimized parameters
    """
    from .config import Config
    
    # Validate bitrate range
    min_bitrate = 32000
    max_bitrate = 320000
    
    if aac_bitrate < min_bitrate:
        logger.warning(f"AAC bitrate {aac_bitrate} too low, using minimum {min_bitrate}")
        aac_bitrate = min_bitrate
    elif aac_bitrate > max_bitrate:
        logger.warning(f"AAC bitrate {aac_bitrate} too high, using maximum {max_bitrate}")
        aac_bitrate = max_bitrate
    
    # Optimize bitrate based on channels
    channels = source_params.get("nb_channels", 2)
    if channels == 1:  # Mono
        recommended_max = 128000
        if aac_bitrate > recommended_max:
            logger.info(f"Reducing AAC bitrate from {aac_bitrate} to {recommended_max} for mono audio")
            aac_bitrate = recommended_max
    
    # Check sample rate compatibility
    sample_rate = source_params.get("sampling_rate", 48000)
    supported_rates = [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000]
    
    if sample_rate not in supported_rates:
        closest_rate = min(supported_rates, key=lambda x: abs(x - sample_rate))
        logger.warning(f"Sample rate {sample_rate}Hz not optimal for AAC, closest supported: {closest_rate}Hz")
    
    return {
        'validated_bitrate': aac_bitrate,
        'channels': channels,
        'sample_rate': sample_rate,
        'use_pyav': Config.aac_enable_pyav_native,
        'fallback_ffmpeg': Config.aac_fallback_to_ffmpeg
    }


def _get_aac_config_for_import():
    """Get AAC configuration parameters for import operations"""
    from .config import Config
    return {
        'bitrate': Config.aac_default_bitrate,
        'use_pyav': Config.aac_enable_pyav_native,
        'fallback_ffmpeg': Config.aac_fallback_to_ffmpeg,
        'quality_preset': Config.aac_quality_preset,
        'memory_limit': Config.aac_memory_limit_mb
    }


def _log_import_performance(start_time: float, audio_file: pathlib.Path, 
                           target_codec: str, **stats):
    """Log performance metrics for import operations"""
    import_time = time.time() - start_time
    file_size_mb = audio_file.stat().st_size / 1024 / 1024
    
    logger.success(
        f"{target_codec.upper()} import completed: "
        f"{audio_file.name} ({file_size_mb:.1f}MB) in {import_time:.2f}s "
        f"({file_size_mb/import_time:.1f} MB/s)"
    )
    
    if 'compression_ratio' in stats:
        logger.info(f"Compression achieved: {stats['compression_ratio']:.1f}x reduction")




def test_aac_integration():
    """
    Quick integration test for AAC functionality
    This can be called during development to verify AAC integration
    """
    try:
        from .config import Config
        from .aac_access import import_aac_to_zarr
        from .aac_index_backend import build_aac_index
        
        logger.info("AAC integration test started...")
        
        # Test configuration
        original_bitrate = Config.aac_default_bitrate
        Config.set(aac_default_bitrate=128000)
        assert Config.aac_default_bitrate == 128000
        Config.set(aac_default_bitrate=original_bitrate)
        
        logger.success("AAC configuration test passed")
        
        # Test imports
        assert import_aac_to_zarr is not None
        assert build_aac_index is not None
        
        logger.success("AAC module imports test passed")
        logger.success("AAC integration test completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"AAC integration test failed: {e}")
        return False




def _get_source_params(input_file: pathlib.Path) -> dict:
    """Ermittelt grundlegende Parameter einer Audiodatei mit ffprobe.
    
    Args:
        input_file: Pfad zur Audiodatei
        
    Returns:
        dict: Dictionary mit den grundlegenden Eigenschaften der Audiodatei
    """
    logger.trace(f"'_get_source_params' requested for file '{input_file.name}'...")
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate:stream=codec_name:stream=sample_fmt:stream=bit_rate:stream=channels",
        "-of", "json", str(input_file)
    ]
    logger.trace(f"Command to read params with ffprobe: '{cmd}'.")
    out = subprocess.check_output(args=cmd)
    info = json.loads(out)
    if ('streams' not in info) or (len(info["streams"]) == 0):
        logger.error(f"ffprobe doesn't found any media stream information in file {input_file.name}.")
        raise ValueError(f"ffprobe doesn't found any media stream information in file {input_file.name}.")
    logger.trace("Reading done. Sort into the dictionary 'source_params'...")
    source_params = {
                "sampling_rate": int(info['streams'][0]['sample_rate']) if 'sample_rate' in info['streams'][0] else None,
                "sample_format": info['streams'][0]['sample_fmt'] if 'sample_fmt' in info['streams'][0] else "s16",
                "bit_rate": int(info['streams'][0]['bit_rate']) if 'bit_rate' in info['streams'][0] else None,
                "nb_channels": int(info['streams'][0]['channels']) if 'channels' in info['streams'][0] else 1
            }
    logger.debug(f"Read parameters in 'source_params' are: {source_params}")
    return source_params



def safe_get_sample_format_dtype(sample_fmt, fallback_fmt="s16"):
    """Safe conversion of sample formats to dtype"""
    try:
        if sample_fmt is None or sample_fmt == "":
            sample_fmt = fallback_fmt
        return AudioSampleFormatMap.get(sample_fmt, AudioSampleFormatMap[fallback_fmt])
    except (KeyError, TypeError):
        return AudioSampleFormatMap[fallback_fmt]
