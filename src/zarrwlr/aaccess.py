def extract_audio_segment(zarr_group, start_sample, end_sample, dtype=np.int16):
    """
    Extrahiert ein Audiosegment aus einer Zarr-Gruppe, unabhängig vom verwendeten Codec.
    
    ENHANCED VERSION (Step 2.0): Added AAC-LC support with format auto-detection.
    
    Args:
        zarr_group: Zarr-Gruppe mit den Audiodaten und dem Index
        start_sample: Erstes Sample, das extrahiert werden soll
        end_sample: Letztes Sample, das extrahiert werden soll
        dtype: Datentyp der Ausgabe (np.int16 oder np.float32)
        
    Returns:
        np.ndarray: Extrahiertes Audiosegment
    """  
    if AUDIO_DATA_BLOB_ARRAY_NAME not in zarr_group:
        raise ValueError("No audio data found in zarr_group")
    
    audio_blob_array = zarr_group[AUDIO_DATA_BLOB_ARRAY_NAME]
    codec = audio_blob_array.attrs.get('codec', 'unknown')

    if codec == 'flac':
        return extract_audio_segment_flac(zarr_group, audio_blob_array, start_sample, end_sample, dtype)
    elif codec == 'aac':
        return extract_audio_segment_aac(zarr_group, audio_blob_array, start_sample, end_sample, dtype)
    else:
        raise ValueError(f"Unsupported codec: {codec}")


def parallel_extract_audio_segments(zarr_group, segments, dtype=np.int16, max_workers=4):
    """
    Extrahiert mehrere Audiosegmente parallel aus einer Zarr-Gruppe.
    
    ENHANCED VERSION (Step 2.0): Added AAC-LC support with format auto-detection.
    """  
    if AUDIO_DATA_BLOB_ARRAY_NAME not in zarr_group:
        raise ValueError("No audio data found in zarr_group")

    audio_blob_array = zarr_group[AUDIO_DATA_BLOB_ARRAY_NAME]
    codec = audio_blob_array.attrs.get('codec', 'unknown')
    
    if codec == 'flac':
        if audio_blob_array is None:
            raise ValueError("FLAC requires legacy audio_data_blob_array format")
        return parallel_extract_audio_segments_flac(zarr_group, audio_blob_array, segments, dtype, max_workers)
        
    elif codec == 'aac':
        if audio_blob_array is None:
            raise ValueError("AAC requires legacy audio_data_blob_array format")
        return parallel_extract_audio_segments_aac(zarr_group, audio_blob_array, segments, dtype, max_workers)
    else:
        raise ValueError(f"Nicht unterstützter Codec: {codec}")
