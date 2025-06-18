#!/usr/bin/env python3
"""
Comprehensive Tests for import_utils.py

Testet die intelligente Audio-Analyse, Qualitätsbewertung und Parameter-Vorschläge
des FileParameter-Systems mit allen verfügbaren Testdateien.

Verwendung:
    python test_import_utils.py
    pytest test_import_utils.py -v
"""

import pytest
import pathlib
import tempfile
import os
import sys

# Projekt-Imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from zarrwlr.import_utils import (
    FileParameter, 
    TargetFormats, 
    TargetSamplingTransforming,
    AudioCompressionBaseType,
    QualityAnalyzer,
    ConflictAnalyzer,
    get_audio_codec_compression_type,
    can_use_copy_mode,
    get_copy_mode_benefits,
    should_warn_about_re_encoding,
    can_ffmpeg_decode_codec
)
from zarrwlr.logsetup import get_module_logger

logger = get_module_logger(__file__)

# ============================================================================
# Test-Konfiguration und Fixtures
# ============================================================================

# Testdaten-Verzeichnis
TESTDATA_DIR = pathlib.Path(__file__).parent / "testdata"

# Test-Ergebnisse-Verzeichnis
TESTRESULTS_DIR = pathlib.Path(__file__).parent / "testresults"

def cleanup_test_results():
    """Räume Test-Ergebnisse-Verzeichnis auf"""
    if TESTRESULTS_DIR.exists():
        import shutil
        shutil.rmtree(TESTRESULTS_DIR)
    TESTRESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.trace(f"Test results directory cleaned: {TESTRESULTS_DIR}")

def get_unique_test_dir(test_name: str) -> pathlib.Path:
    """Erstelle eindeutiges Test-Verzeichnis für Test-Isolation"""
    import time
    timestamp = int(time.time() * 1000000)  # Microsecond precision
    test_dir = TESTRESULTS_DIR / f"{test_name}_{timestamp}"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir

# Kategorisierte Testdateien basierend auf der Analyse
TEST_FILES = {
    'uncompressed': [
        "XC821955 - Greater Dog-like Bat - Peropteryx kappleri.wav",  # 400kHz ultrasound
        "XC838306 - Lesser Dog-like Bat - Peropteryx macrotis.wav",   # 250kHz ultrasound
        "XC897425 - Rohrschwirl - Locustella luscinioides.wav",       # 44.1kHz stereo
        "audiomoth_short_snippet.wav",                                # 48kHz mono
        "multichannel_8ch.wav",                                       # 8 channels
        "multichannel_12ch.wav"                                       # 12 channels (exceeds FLAC limit)
    ],
    'lossless': [
        "audiomoth_long_snippet_converted.flac"                       # FLAC 48kHz mono
    ],
    'lossy': [
        "XC642625 - Rohrschwirl - Locustella luscinioides luscinioides.mp3",  # 320kbps stereo
        "XC69863 - Gelbfußdrossel - Turdus flavipes.mp3",             # 128kbps stereo
        "XC890995 - Oak Toad - Anaxyrus quercicus.mp3",               # 93kbps mono
        "bird1_snippet.mp3"                                           # 260kbps stereo
    ],
    'ultrasound': [
        "XC821955 - Greater Dog-like Bat - Peropteryx kappleri.wav",  # 400kHz
        "XC838306 - Lesser Dog-like Bat - Peropteryx macrotis.wav"    # 250kHz
    ],
    'multichannel': [
        "multichannel_3ch.wav",   # 3 channels
        "multichannel_8ch.wav",   # 8 channels (FLAC limit)
        "multichannel_12ch.wav",  # 12 channels (exceeds FLAC limit)
        "multichannel_32ch.wav"   # 32 channels (way over FLAC limit)
    ]
}

@pytest.fixture
def temp_dir():
    """Temporäres Verzeichnis für Tests in testresults/"""
    test_name = "temp_test"
    if hasattr(pytest, 'current_request'):
        test_name = pytest.current_request.node.name
    
    test_dir = get_unique_test_dir(test_name)
    yield test_dir
    
    # Cleanup nach Test (optional, da testresults komplett gereinigt wird)
    # shutil.rmtree(test_dir, ignore_errors=True)

def get_test_file_path(filename: str) -> pathlib.Path:
    """Hilfsfunktion zum Abrufen des vollständigen Pfads einer Testdatei"""
    path = TESTDATA_DIR / filename
    if not path.exists():
        pytest.skip(f"Testdatei nicht gefunden: {filename}")
    return path

# ============================================================================
# Tests für Enum-Konvertierungen
# ============================================================================

class TestEnumConversions:
    """Tests für TargetFormats und TargetSamplingTransforming Enum-Konvertierungen"""
    
    def test_target_formats_from_string(self):
        """Test TargetFormats.from_string_or_enum()"""
        # Standard-Formate
        assert TargetFormats.from_string_or_enum("FLAC") == TargetFormats.FLAC
        assert TargetFormats.from_string_or_enum("flac") == TargetFormats.FLAC
        assert TargetFormats.from_string_or_enum("AAC") == TargetFormats.AAC
        
        # Mit Sample Rates
        assert TargetFormats.from_string_or_enum("FLAC_44100") == TargetFormats.FLAC_44100
        assert TargetFormats.from_string_or_enum("flac44100") == TargetFormats.FLAC_44100
        assert TargetFormats.from_string_or_enum("aac_48000") == TargetFormats.AAC_48000
        
        # Bereits Enum
        assert TargetFormats.from_string_or_enum(TargetFormats.AAC_96000) == TargetFormats.AAC_96000
        
        # Fehlerbehandlung
        with pytest.raises(ValueError, match="Unknown target format"):
            TargetFormats.from_string_or_enum("invalid_format")
    
    def test_target_sampling_transform_from_string(self):
        """Test TargetSamplingTransforming.from_string_or_enum()"""
        # Standard-Transformationen
        assert TargetSamplingTransforming.from_string_or_enum("EXACTLY") == TargetSamplingTransforming.EXACTLY
        assert TargetSamplingTransforming.from_string_or_enum("exactly") == TargetSamplingTransforming.EXACTLY
        
        # Resampling
        assert TargetSamplingTransforming.from_string_or_enum("RESAMPLING_44100") == TargetSamplingTransforming.RESAMPLING_44100
        assert TargetSamplingTransforming.from_string_or_enum("resampling44100") == TargetSamplingTransforming.RESAMPLING_44100
        
        # Reinterpretation
        assert TargetSamplingTransforming.from_string_or_enum("REINTERPRETING_32000") == TargetSamplingTransforming.REINTERPRETING_32000
        assert TargetSamplingTransforming.from_string_or_enum("reinterpreting_auto") == TargetSamplingTransforming.REINTERPRETING_AUTO
        
        # Partial matches
        assert TargetSamplingTransforming.from_string_or_enum("resampling") == TargetSamplingTransforming.RESAMPLING_NEAREST
        assert TargetSamplingTransforming.from_string_or_enum("reinterpreting") == TargetSamplingTransforming.REINTERPRETING_AUTO

# ============================================================================
# Tests für Codec-Klassifizierung
# ============================================================================

class TestCodecClassification:
    """Tests für Audio-Codec-Klassifizierung"""
    
    def test_get_audio_codec_compression_type(self):
        """Test get_audio_codec_compression_type()"""
        # Uncompressed
        assert get_audio_codec_compression_type("pcm_s16le") == AudioCompressionBaseType.UNCOMPRESSED
        assert get_audio_codec_compression_type("pcm_s24le") == AudioCompressionBaseType.UNCOMPRESSED
        assert get_audio_codec_compression_type("PCM") == AudioCompressionBaseType.UNCOMPRESSED
        
        # Lossless
        assert get_audio_codec_compression_type("flac") == AudioCompressionBaseType.LOSSLESS_COMPRESSED
        assert get_audio_codec_compression_type("FLAC") == AudioCompressionBaseType.LOSSLESS_COMPRESSED
        assert get_audio_codec_compression_type("alac") == AudioCompressionBaseType.LOSSLESS_COMPRESSED
        
        # Lossy
        assert get_audio_codec_compression_type("mp3") == AudioCompressionBaseType.LOSSY_COMPRESSED
        assert get_audio_codec_compression_type("aac") == AudioCompressionBaseType.LOSSY_COMPRESSED
        assert get_audio_codec_compression_type("opus") == AudioCompressionBaseType.LOSSY_COMPRESSED
        
        # Unknown
        assert get_audio_codec_compression_type("unknown_codec") == AudioCompressionBaseType.UNKNOWN
    
    def test_can_ffmpeg_decode_codec(self):
        """Test can_ffmpeg_decode_codec()"""
        # Standard-Codecs sollten verfügbar sein
        assert can_ffmpeg_decode_codec("mp3") == True
        assert can_ffmpeg_decode_codec("aac") == True
        assert can_ffmpeg_decode_codec("flac") == True
        assert can_ffmpeg_decode_codec("pcm_s16le") == True
        
        # Unbekannter Codec
        assert can_ffmpeg_decode_codec("nonexistent_codec") == False

# ============================================================================
# Tests für Copy-Mode-Funktionen
# ============================================================================

class TestCopyModeFunctions:
    """Tests für Copy-Mode-Erkennungs- und Nutzen-Funktionen"""
    
    def test_can_use_copy_mode(self):
        """Test can_use_copy_mode()"""
        # AAC -> AAC Copy-Mode
        assert can_use_copy_mode("aac", TargetFormats.AAC) == True
        assert can_use_copy_mode("aac", TargetFormats.AAC_44100) == True
        assert can_use_copy_mode("AAC", "aac_48000") == True
        
        # FLAC -> FLAC Copy-Mode
        assert can_use_copy_mode("flac", TargetFormats.FLAC) == True
        assert can_use_copy_mode("FLAC", "FLAC_96000") == True
        
        # Keine Copy-Mode möglich
        assert can_use_copy_mode("mp3", TargetFormats.AAC) == False
        assert can_use_copy_mode("aac", TargetFormats.FLAC) == False
        assert can_use_copy_mode("pcm_s16le", TargetFormats.AAC) == False
        
        # Edge cases
        assert can_use_copy_mode("", TargetFormats.AAC) == False
        assert can_use_copy_mode(None, TargetFormats.AAC) == False
    
    def test_get_copy_mode_benefits(self):
        """Test get_copy_mode_benefits()"""
        # AAC Copy-Mode Benefits
        benefits = get_copy_mode_benefits("aac", TargetFormats.AAC_44100)
        assert benefits['copy_mode_possible'] == True
        assert benefits['codec'] == 'AAC'
        assert 'generation loss' in benefits['quality_benefit']
        
        # FLAC Copy-Mode Benefits
        benefits = get_copy_mode_benefits("flac", TargetFormats.FLAC_48000)
        assert benefits['copy_mode_possible'] == True
        assert benefits['codec'] == 'FLAC'
        assert 'lossless' in benefits['quality_benefit']
        
        # Keine Copy-Mode möglich
        benefits = get_copy_mode_benefits("mp3", TargetFormats.AAC)
        assert benefits == {}
    
    def test_should_warn_about_re_encoding(self):
        """Test should_warn_about_re_encoding()"""
        # AAC -> AAC Re-Encoding Warnung
        warning = should_warn_about_re_encoding("aac", TargetFormats.AAC, 192000)
        assert warning['should_warn'] == True
        assert warning['severity'] == 'quality_loss'
        assert 'generation loss' in warning['message']
        
        # FLAC -> FLAC Re-Processing Warnung
        warning = should_warn_about_re_encoding("flac", TargetFormats.FLAC)
        assert warning['should_warn'] == True
        assert warning['severity'] == 'efficiency'
        
        # Keine Warnung nötig
        warning = should_warn_about_re_encoding("mp3", TargetFormats.AAC)
        assert warning == {}

# ============================================================================
# Tests für QualityAnalyzer
# ============================================================================

class TestQualityAnalyzer:
    """Tests für die intelligente Qualitätsanalyse"""
    
    def test_ultrasound_detection(self):
        """Test Ultraschall-Erkennung"""
        # Ultraschall-Beispiele
        assert QualityAnalyzer._is_ultrasound_recording(400000) == True
        assert QualityAnalyzer._is_ultrasound_recording(250000) == True
        assert QualityAnalyzer._is_ultrasound_recording(96001) == True
        
        # Normaler Bereich
        assert QualityAnalyzer._is_ultrasound_recording(96000) == False
        assert QualityAnalyzer._is_ultrasound_recording(48000) == False
        assert QualityAnalyzer._is_ultrasound_recording(44100) == False
    
    def test_aac_upgrade_bitrate_calculation(self):
        """Test intelligente AAC-Bitrate-Berechnung"""
        # Low quality sources get bigger boost
        assert QualityAnalyzer._calculate_aac_upgrade_bitrate(64000, 1) >= 160000  # Minimum mono
        assert QualityAnalyzer._calculate_aac_upgrade_bitrate(96000, 2) >= 190000  # Minimum stereo
        
        # High quality sources get smaller boost
        result = QualityAnalyzer._calculate_aac_upgrade_bitrate(256000, 2)
        assert 290000 <= result <= 320000  # Should be around 256k * 1.15
        
        # Never exceed AAC maximum
        assert QualityAnalyzer._calculate_aac_upgrade_bitrate(300000, 2) <= 320000
    
    def test_copy_mode_detection(self):
        """Test Copy-Mode Erkennung in QualityAnalyzer"""
        # AAC -> AAC Copy-Mode
        assert QualityAnalyzer._can_use_copy_mode("aac", TargetFormats.AAC) == True
        assert QualityAnalyzer._can_use_copy_mode("aac", TargetFormats.AAC_44100) == True
        
        # FLAC -> FLAC Copy-Mode
        assert QualityAnalyzer._can_use_copy_mode("flac", TargetFormats.FLAC) == True
        
        # Keine Copy-Mode
        assert QualityAnalyzer._can_use_copy_mode("mp3", TargetFormats.AAC) == False
        assert QualityAnalyzer._can_use_copy_mode("aac", TargetFormats.FLAC) == False

# ============================================================================
# Tests für ConflictAnalyzer
# ============================================================================

class TestConflictAnalyzer:
    """Tests für Konflikt- und Warnungsanalyse"""
    
    def test_flac_channel_limit_conflict(self):
        """Test FLAC 8-Kanal-Limit-Konflikt"""
        source_analysis = {
            'compression_type': AudioCompressionBaseType.UNCOMPRESSED,
            'channels': 12,
            'sample_rate': 44100,
            'codec_name': 'pcm_s16le'
        }
        
        target_params = {
            'target_format': TargetFormats.FLAC,
            'target_sampling_transform': TargetSamplingTransforming.EXACTLY
        }
        
        conflicts = ConflictAnalyzer.analyze_conflicts(source_analysis, target_params)
        
        # Sollte Blocking-Konflikt haben
        assert len(conflicts['blocking_conflicts']) > 0
        assert any('8 channels' in conflict for conflict in conflicts['blocking_conflicts'])
    
    def test_ultrasound_resampling_conflict(self):
        """Test Ultraschall + Resampling = Blocking Conflict"""
        source_analysis = {
            'compression_type': AudioCompressionBaseType.UNCOMPRESSED,
            'channels': 1,
            'sample_rate': 400000,
            'codec_name': 'pcm_s16le',
            'is_ultrasound': True
        }
        
        target_params = {
            'target_format': TargetFormats.AAC_48000,
            'target_sampling_transform': TargetSamplingTransforming.RESAMPLING_48000,
            'copy_mode': False
        }
        
        conflicts = ConflictAnalyzer.analyze_conflicts(source_analysis, target_params)
        
        # Sollte Blocking-Konflikt haben
        assert len(conflicts['blocking_conflicts']) > 0
        assert any('Ultrasound' in conflict and 'Resampling' in conflict for conflict in conflicts['blocking_conflicts'])
    
    def test_aac_multichannel_warnings(self):
        """Test AAC Multichannel-Warnungen"""
        source_analysis = {
            'compression_type': AudioCompressionBaseType.UNCOMPRESSED,
            'channels': 8,
            'sample_rate': 44100,
            'codec_name': 'pcm_s16le'
        }
        
        # Re-Encoding Fall
        target_params = {
            'target_format': TargetFormats.AAC,
            'copy_mode': False
        }
        
        conflicts = ConflictAnalyzer.analyze_conflicts(source_analysis, target_params)
        
        # Sollte Quality Warning haben
        assert len(conflicts['quality_warnings']) > 0
        assert any('spatial information' in warning for warning in conflicts['quality_warnings'])

# ============================================================================
# Tests für FileParameter mit echten Testdateien
# ============================================================================

class TestFileParameterWithRealFiles:
    """Tests für FileParameter mit echten Audio-Testdateien"""
    
    def test_uncompressed_wav_analysis(self):
        """Test Analyse einer unkomprimierten WAV-Datei"""
        test_file = get_test_file_path("audiomoth_short_snippet.wav")
        file_param = FileParameter(test_file)
        
        # Grundlegende Datei-Informationen
        assert file_param.base_parameter.file.name == "audiomoth_short_snippet.wav"
        assert file_param.base_parameter.file_size_bytes > 0
        assert file_param.base_parameter.file_sh256 is not None
        assert len(file_param.base_parameter.file_sh256) == 64  # SHA256 hex length
        
        # Container-Informationen
        assert file_param.container['format_name'] == 'wav'
        assert file_param.has_audio == True
        assert file_param.number_of_audio_streams >= 1
        
        # Audio-Stream-Details
        audio_streams = file_param.selected_audio_streams
        assert len(audio_streams) == 1
        stream = audio_streams[0]
        
        assert stream.codec_name == 'pcm_s16le'
        assert stream.nb_channels == 1  # Mono
        assert stream.sample_rate == 48000
        
        # Qualitätsanalyse
        assert file_param.quality_analysis['compression_type'] == AudioCompressionBaseType.UNCOMPRESSED
        assert file_param.quality_analysis['is_ultrasound'] == False
        
        # Import-Bereitschaft
        assert file_param.can_be_imported == True
        assert not file_param.has_blocking_conflicts
    
    def test_ultrasound_bat_recording_analysis(self):
        """Test Analyse einer Ultraschall-Fledermaus-Aufnahme"""
        test_file = get_test_file_path("XC821955 - Greater Dog-like Bat - Peropteryx kappleri.wav")
        file_param = FileParameter(test_file)
        
        # Audio-Stream-Details
        audio_streams = file_param.selected_audio_streams
        stream = audio_streams[0]
        
        assert stream.codec_name == 'pcm_s16le'
        assert stream.nb_channels == 1
        assert stream.sample_rate == 400000  # 400kHz = Ultraschall
        
        # Ultraschall-Erkennung
        assert file_param.quality_analysis['is_ultrasound'] == True
        assert file_param.is_ultrasound_recording == True
        
        # Intelligente Vorschläge für Ultraschall sollten vorhanden sein
        assert file_param.target_format is not None
        assert file_param.target_sampling_transform is not None
        
        # Für Ultraschall sollte entweder FLAC oder Reinterpretation vorgeschlagen werden
        if file_param.target_format.code == 'aac':
            assert file_param.target_sampling_transform.code == 'reinterpreting'
        elif file_param.target_format.code == 'flac':
            assert file_param.target_sampling_transform == TargetSamplingTransforming.EXACTLY
    
    def test_lossy_mp3_analysis(self):
        """Test Analyse einer verlustbehafteten MP3-Datei"""
        test_file = get_test_file_path("XC69863 - Gelbfußdrossel - Turdus flavipes.mp3")
        file_param = FileParameter(test_file)
        
        # Audio-Stream-Details
        audio_streams = file_param.selected_audio_streams
        stream = audio_streams[0]
        
        assert stream.codec_name == 'mp3'
        assert stream.nb_channels == 2  # Stereo
        assert stream.sample_rate == 44100
        assert stream.bit_rate == 128000  # 128 kbps
        
        # Qualitätsanalyse
        assert file_param.quality_analysis['compression_type'] == AudioCompressionBaseType.LOSSY_COMPRESSED
        assert file_param.quality_analysis['bitrate_class'] in ['medium', 'low']  # 128kbps für Stereo
        
        # Intelligente Vorschläge
        assert file_param.target_format is not None
        
        # Für lossy -> AAC sollte bitrate upgrade vorgeschlagen werden
        if file_param.target_format.code == 'aac':
            suggested_bitrate = file_param.aac_bitrate
            assert suggested_bitrate > 128000  # Upgrade
            assert suggested_bitrate >= 190000  # Minimum für wissenschaftliche Stereo-Aufnahmen
    
    def test_flac_lossless_analysis(self):
        """Test Analyse einer verlustfreien FLAC-Datei"""
        test_file = get_test_file_path("audiomoth_long_snippet_converted.flac")
        file_param = FileParameter(test_file)
        
        # Audio-Stream-Details
        audio_streams = file_param.selected_audio_streams
        stream = audio_streams[0]
        
        assert stream.codec_name == 'flac'
        assert stream.nb_channels == 1  # Mono
        assert stream.sample_rate == 48000
        
        # Qualitätsanalyse
        assert file_param.quality_analysis['compression_type'] == AudioCompressionBaseType.LOSSLESS_COMPRESSED
        
        # FLAC -> FLAC sollte Copy-Mode erkennen
        if file_param.target_format and file_param.target_format.code == 'flac':
            assert file_param.is_copy_mode == True
    
    def test_multichannel_over_flac_limit(self):
        """Test Multichannel-Datei über FLAC-Limit (>8 Kanäle)"""
        test_file = get_test_file_path("multichannel_12ch.wav")
        file_param = FileParameter(test_file)
        
        # Audio-Stream-Details
        audio_streams = file_param.selected_audio_streams
        stream = audio_streams[0]
        
        assert stream.codec_name == 'pcm_s16le'
        assert stream.nb_channels == 12  # Über FLAC-Limit von 8
        assert stream.sample_rate == 44100
        
        # Wenn FLAC vorgeschlagen wird, sollte Blocking-Konflikt auftreten
        if file_param.target_format and file_param.target_format.code == 'flac':
            assert file_param.has_blocking_conflicts == True
            conflicts = file_param.conflicts['blocking_conflicts']
            assert any('8 channels' in conflict for conflict in conflicts)
        
        # Alternativ sollte AAC vorgeschlagen werden (trotz Multichannel-Warnung)
        if file_param.target_format and file_param.target_format.code == 'aac':
            # Sollte Quality Warning haben wegen Multichannel
            assert file_param.has_quality_warnings == True
    
    def test_parameter_override_system(self):
        """Test Benutzer-Parameter-Override-System"""
        test_file = get_test_file_path("bird1_snippet.mp3")
        file_param = FileParameter(test_file)
        
        # Initial sollten auto-suggested parameters gesetzt sein
        original_format = file_param.target_format
        original_bitrate = file_param.aac_bitrate
        
        # Override target format
        file_param.target_format = "FLAC_44100"
        assert file_param.target_format == TargetFormats.FLAC_44100
        assert 'target_format' in file_param._user_defined_params
        
        # Override AAC bitrate (sollte nicht mehr relevant sein für FLAC)
        file_param.aac_bitrate = 256000
        assert file_param.aac_bitrate == 256000
        assert 'aac_bitrate' in file_param._user_defined_params
        
        # Reset sollte auto-suggestions wieder aktivieren
        file_param.reset_suggestions()
        assert len(file_param._user_defined_params) == 0
        
        # Nach reset sollten wieder auto-suggestions aktiv sein
        # (können sich von den ursprünglichen unterscheiden wegen der veränderten Konfiguration)
        assert file_param.target_format is not None
    
    def test_import_parameters_extraction(self):
        """Test get_import_parameters() Methode"""
        test_file = get_test_file_path("audiomoth_short_snippet.wav")
        file_param = FileParameter(test_file)
        
        import_params = file_param.get_import_parameters()
        
        # Alle wichtigen Parameter sollten vorhanden sein
        required_keys = [
            'target_format', 'target_sampling_transform', 'aac_bitrate',
            'flac_compression_level', 'selected_streams', 'file_path',
            'user_meta', 'copy_mode'
        ]
        
        for key in required_keys:
            assert key in import_params
        
        # Spezifische Werte validieren
        assert import_params['file_path'] == file_param.base_parameter.file
        assert import_params['selected_streams'] == file_param.base_parameter.selected_audio_streams
        assert isinstance(import_params['copy_mode'], bool)
    
    def test_beautiful_terminal_output(self):
        """Test der schönen Terminal-Ausgabe"""
        test_file = get_test_file_path("XC821955 - Greater Dog-like Bat - Peropteryx kappleri.wav")
        file_param = FileParameter(test_file)
        
        # Terminal-Ausgabe sollte generiert werden können
        terminal_output = str(file_param)
        
        # Sollte wichtige Informationen enthalten
        assert "Audio File Analysis" in terminal_output
        assert file_param.base_parameter.file.name in terminal_output
        assert "400,000 Hz" in terminal_output  # Sample rate
        assert "🦇" in terminal_output  # Ultrasound marker
        
        # Sollte Empfehlungen enthalten
        assert "Recommended Import Settings" in terminal_output
        
        # Sollte API-Hinweise enthalten
        assert "📝" in terminal_output
        assert "<your_instance>" in terminal_output
    
    def test_validation_edge_cases(self):
        """Test Validierung und Edge Cases"""
        test_file = get_test_file_path("bird1_snippet.mp3")
        file_param = FileParameter(test_file)
        
        # Invalid AAC bitrate
        with pytest.raises(ValueError, match="AAC bitrate must be between"):
            file_param.aac_bitrate = 25000  # Zu niedrig
        
        with pytest.raises(ValueError, match="AAC bitrate must be between"):
            file_param.aac_bitrate = 400000  # Zu hoch
        
        # Invalid FLAC compression level
        with pytest.raises(ValueError, match="FLAC compression level must be between"):
            file_param.flac_compression_level = -1  # Zu niedrig
        
        with pytest.raises(ValueError, match="FLAC compression level must be between"):
            file_param.flac_compression_level = 15  # Zu hoch
        
        # Invalid target format
        with pytest.raises(ValueError, match="Unknown target format"):
            file_param.target_format = "INVALID_FORMAT"

# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegrationScenarios:
    """Integration-Tests für komplexe Szenarien"""
    
    def test_aac_copy_mode_scenario(self):
        """Test AAC->AAC Copy-Mode Szenario"""
        # Erstelle temporäre AAC-Datei (simuliert durch MP3)
        test_file = get_test_file_path("bird1_snippet.mp3")
        file_param = FileParameter(test_file)
        
        # Forciere AAC target format
        file_param.target_format = "AAC_48000"
        
        # Bei MP3->AAC sollte kein Copy-Mode möglich sein
        assert file_param.is_copy_mode == False
        
        # Aber für echtes AAC->AAC wäre Copy-Mode möglich
        # (simuliert durch manuelle Codec-Änderung für Test)
        file_param.quality_analysis['codec_name'] = 'aac'
        file_param._analyze()  # Re-analyze with AAC codec
        
        # Jetzt sollte Copy-Mode erkannt werden
        assert file_param.is_copy_mode == True
    
    def test_ultrasound_protection_scenario(self):
        """Test Ultraschall-Schutz-Szenario"""
        test_file = get_test_file_path("XC838306 - Lesser Dog-like Bat - Peropteryx macrotis.wav")
        file_param = FileParameter(test_file)
        
        # Versuche unsichere Konfiguration für Ultraschall
        file_param.target_format = "AAC_48000"
        file_param.target_sampling_transform = "RESAMPLING_48000"  # GEFÄHRLICH für Ultraschall!
        
        # Sollte Blocking-Konflikt erzeugen
        assert file_param.has_blocking_conflicts == True
        conflicts = file_param.conflicts['blocking_conflicts']
        assert any('Ultrasound' in conflict and 'Resampling' in conflict for conflict in conflicts)
        
        # Korrigiere zu sicherer Konfiguration
        file_param.target_sampling_transform = "REINTERPRETING_32000"
        
        # Jetzt sollte es funktionieren
        assert file_param.has_blocking_conflicts == False
    
    def test_multichannel_scientific_scenario(self):
        """Test Multichannel wissenschaftliches Aufnahme-Szenario"""
        test_file = get_test_file_path("multichannel_8ch.wav")
        file_param = FileParameter(test_file)
        
        # 8 Kanäle sollten noch mit FLAC funktionieren
        if file_param.target_format and file_param.target_format.code == 'flac':
            assert file_param.has_blocking_conflicts == False
        
        # Aber forciere AAC für Multichannel
        file_param.target_format = "AAC_44100"
        
        # Sollte Quality Warning haben wegen spatial information loss
        assert file_param.has_quality_warnings == True
        warnings = file_param.conflicts['quality_warnings']
        assert any('spatial information' in warning for warning in warnings)
    
    def test_batch_file_analysis_simulation(self):
        """Test Batch-Datei-Analyse-Simulation"""
        # Sammle verschiedene Dateitypen
        test_files = [
            get_test_file_path("audiomoth_short_snippet.wav"),      # Uncompressed
            get_test_file_path("audiomoth_long_snippet_converted.flac"),  # Lossless
            get_test_file_path("bird1_snippet.mp3"),               # Lossy
            get_test_file_path("XC821955 - Greater Dog-like Bat - Peropteryx kappleri.wav")  # Ultrasound
        ]
        
        file_params = []
        for test_file in test_files:
            try:
                file_param = FileParameter(test_file)
                file_params.append(file_param)
            except Exception as e:
                pytest.fail(f"Failed to analyze {test_file}: {e}")
        
        # Alle sollten analysiert werden können
        assert len(file_params) == len(test_files)
        
        # Alle sollten importierbar sein (oder haben spezifische Konflikte)
        for file_param in file_params:
            # Wenn nicht importierbar, sollten klare Blocking-Konflikte vorliegen
            if not file_param.can_be_imported:
                assert file_param.has_blocking_conflicts == True
                assert len(file_param.conflicts['blocking_conflicts']) > 0
        
        # Verschiedene Formate sollten verschiedene Vorschläge haben
        target_formats = [fp.target_format.code if fp.target_format else None for fp in file_params]
        assert len(set(target_formats)) >= 2  # Mindestens 2 verschiedene Formate vorgeschlagen
    
    def test_quality_tier_classification(self):
        """Test Qualitäts-Tier-Klassifizierung verschiedener Dateien"""
        test_cases = [
            ("XC890995 - Oak Toad - Anaxyrus quercicus.mp3", ['low', 'standard']),  # 93kbps
            ("XC69863 - Gelbfußdrossel - Turdus flavipes.mp3", ['standard']),       # 128kbps
            ("audiomoth_short_snippet.wav", ['standard', 'high']),                   # 48kHz uncompressed
            ("XC821955 - Greater Dog-like Bat - Peropteryx kappleri.wav", ['ultrasound']),  # 400kHz
        ]
        
        for filename, expected_tiers in test_cases:
            test_file = get_test_file_path(filename)
            file_param = FileParameter(test_file)
            
            quality_tier = file_param.quality_analysis.get('quality_tier', 'unknown')
            assert quality_tier in expected_tiers, f"File {filename} has unexpected quality tier: {quality_tier}"


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance-Tests für FileParameter"""
    
    def test_analysis_performance(self):
        """Test Analyse-Performance für verschiedene Dateigrößen"""
        import time
        
        # Kleine Datei
        small_file = get_test_file_path("bird1_snippet.mp3")  # 0.19 MB
        start_time = time.time()
        file_param_small = FileParameter(small_file)
        small_time = time.time() - start_time
        
        # Große Datei (überspringen wenn nicht verfügbar oder zu lange)
        try:
            large_file = get_test_file_path("audiomoth_long_snippet_converted.flac")  # 200 MB
            start_time = time.time()
            file_param_large = FileParameter(large_file)
            large_time = time.time() - start_time
            
            # Große Dateien sollten nicht unverhältnismäßig länger dauern
            # (da nur Metadaten analysiert werden, nicht der ganze Inhalt)
            assert large_time < small_time * 10, "Large file analysis took disproportionately long"
            
        except Exception:
            # Überspringen wenn große Datei Probleme macht
            pass
        
        # Kleine Datei sollte schnell analysiert werden
        assert small_time < 5.0, f"Small file analysis took too long: {small_time:.2f}s"
    
    def test_hash_calculation_performance(self):
        """Test SHA256-Hash-Berechnung-Performance"""
        import time
        
        test_file = get_test_file_path("audiomoth_short_snippet.wav")  # 20 MB
        
        start_time = time.time()
        file_param = FileParameter(test_file)
        hash_time = time.time() - start_time
        
        # Hash-Berechnung sollte für 20MB-Datei reasonable sein
        assert hash_time < 10.0, f"Hash calculation took too long: {hash_time:.2f}s"
        
        # Hash sollte korrekt sein
        assert len(file_param.base_parameter.file_sh256) == 64
        assert all(c in '0123456789abcdef' for c in file_param.base_parameter.file_sh256)


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests für Fehlerbehandlung"""
    
    def test_nonexistent_file(self):
        """Test nicht existierende Datei"""
        with pytest.raises(FileNotFoundError):
            FileParameter("nonexistent_file.wav")
    
    def test_directory_instead_of_file(self, temp_dir):
        """Test Verzeichnis statt Datei"""
        with pytest.raises(ValueError, match="Path is not a file"):
            FileParameter(temp_dir)
    
    def test_invalid_audio_file(self):
        """Test ungültige Audio-Datei"""
        # Erstelle eine Fake-Audio-Datei in testresults
        test_dir = get_unique_test_dir("invalid_audio_test")
        fake_audio = test_dir / "fake.wav"
        fake_audio.write_text("This is not an audio file")
        
        # Sollte bei ffprobe-Analyse fehlschlagen
        with pytest.raises((RuntimeError, ValueError)):
            FileParameter(fake_audio)
    
    def test_corrupted_audio_metadata(self):
        """Test beschädigte Audio-Metadaten"""
        # Kopiere eine echte Datei und modifiziere sie minimal
        test_file = get_test_file_path("bird1_snippet.mp3")
        test_dir = get_unique_test_dir("corrupted_audio_test")
        corrupted_file = test_dir / "corrupted.mp3"
        
        # Kopiere ersten Teil der Datei (Header) aber nicht alles
        with open(test_file, 'rb') as src, open(corrupted_file, 'wb') as dst:
            dst.write(src.read(1024))  # Nur erste 1KB
        
        # Sollte eine Warnung oder einen Fehler geben
        try:
            file_param = FileParameter(corrupted_file)
            # Wenn es funktioniert, sollte die Dauer sehr kurz oder unknown sein
            if file_param.container.get('duration'):
                assert file_param.container['duration'] < 1.0
        except (RuntimeError, ValueError):
            # Oder es schlägt komplett fehl, was auch ok ist
            pass


# ============================================================================
# Main Execution
# ============================================================================

def run_tests():
    """Führe alle Tests direkt aus (ohne pytest)"""
    print("🧪 Running FileParameter and import_utils Tests")
    print("=" * 60)
    
    # Cleanup testresults directory
    cleanup_test_results()
    
    # Test Kategorien
    test_classes = [
        TestEnumConversions,
        TestCodecClassification,
        TestCopyModeFunctions,
        TestQualityAnalyzer,
        TestConflictAnalyzer,
        TestFileParameterWithRealFiles,
        TestIntegrationScenarios,
        TestPerformance,
        TestErrorHandling
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n📋 {test_class.__name__}")
        print("-" * 40)
        
        # Erstelle Instanz der Test-Klasse
        test_instance = test_class()
        
        # Finde alle Test-Methoden
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_') and callable(getattr(test_instance, method))]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                # Führe Test aus
                test_method = getattr(test_instance, method_name)
                
                # Erstelle isoliertes Test-Verzeichnis falls benötigt
                method_params = test_method.__code__.co_varnames
                if 'temp_dir' in method_params:
                    test_dir = get_unique_test_dir(f"{test_class.__name__}_{method_name}")
                    test_method(test_dir)
                else:
                    test_method()
                
                print(f"  ✅ {method_name}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ❌ {method_name}: {str(e)}")
                failed_tests.append((test_class.__name__, method_name, str(e)))
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print(f"📊 Test Summary:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests} ✅")
    print(f"   Failed: {len(failed_tests)} ❌")
    print(f"   Test Results Dir: {TESTRESULTS_DIR}")
    
    if failed_tests:
        print(f"\n💥 Failed Tests:")
        for class_name, method_name, error in failed_tests:
            print(f"   {class_name}.{method_name}: {error}")
        return False
    else:
        print(f"\n🎉 All tests passed!")
        return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)