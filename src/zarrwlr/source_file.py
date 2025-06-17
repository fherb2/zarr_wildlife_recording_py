from dataclasses import dataclass
import subprocess
import json
import pathlib
import hashlib
from typing import Dict, List, Any, Optional, Set
from .utils import safe_int_conversion, safe_float_conversion, can_ffmpeg_decode_codec
from .audio_coding import TargetFormats, TargetSamplingTransforming, AudioCompressionBaseType, get_audio_codec_compression_type

# import and initialize logging
from .logsetup import get_module_logger
logger = get_module_logger(__file__)
logger.trace("Enhanced FileParameter module loading...")

# ############################################################
# ############################################################
#
# Class AudioStreamParameters
# ===========================
# 
# Main audio stream parameters for import processing
#
@dataclass
class AudioStreamParameters:
    """Main parameters of audio streams"""
    index_of_stream_in_file: int
    nb_channels: int = 0
    sample_rate: int|None = None
    sample_format: str|None = None
    codec_name: str|None = None
    nb_samples: int|None = None
    bit_rate: int|None = None  # Bitrate for quality analysis only
#
# End of Class AudioStreamParameters
#
# ############################################################
# ############################################################


# ############################################################
# ############################################################
#
# Class FileBaseParameters
# ========================
# 
# Main audio file parameters for import processing
#
@dataclass
class FileBaseParameters:
    """Main audio file parameters for import processing."""
    file: pathlib.Path = None
    file_size_bytes: int|None = None
    file_sh256: str|None = None
    container_format_name: str|None = None
    stream_parameters: list[AudioStreamParameters] = None
    selected_audio_streams: list[int] = None
    total_nb_of_channels = 0
    total_nb_of_channels_to_import = 0
    
    def __post_init__(self):
        if self.stream_parameters is None:
            self.stream_parameters = []
        if self.selected_audio_streams is None:
            self.selected_audio_streams = []
#
# End of Class FileBaseParameters
#
# ############################################################
# ############################################################


# ############################################################
# ############################################################
#
# Class QualityAnalyzer
# =====================
#
# Analyzes audio quality and suggests optimal parameters
#
class QualityAnalyzer:
    """Analyzes audio quality and provides intelligent parameter suggestions"""
    
    # Ultraschall-Konstanten
    ULTRASOUND_THRESHOLD = 96000  # 96kS/s als Grenze für Ultraschall
    ULTRASOUND_REINTERPRET_TARGET = 32000  # 32kS/s für Reinterpretation
    
    @staticmethod
    def _is_ultrasound_recording(sample_rate: int) -> bool:
        """Prüft, ob es sich um eine Ultraschallaufnahme handelt"""
        return sample_rate > QualityAnalyzer.ULTRASOUND_THRESHOLD
    
    @staticmethod
    def analyze_source_quality(stream_params: AudioStreamParameters) -> dict:
        """
        Analyzes the quality characteristics of the source audio
        
        Returns:
            dict: Quality analysis with compression type, bitrate class, etc.
        """
        codec_compression = get_audio_codec_compression_type(stream_params.codec_name)
        
        analysis = {
            'compression_type': codec_compression,
            'codec_name': stream_params.codec_name,
            'sample_rate': stream_params.sample_rate,
            'bit_rate': stream_params.bit_rate,
            'channels': stream_params.nb_channels,
            'sample_format': stream_params.sample_format,
            'bitrate_class': 'unknown',
            'quality_tier': 'unknown',
            'is_ultrasound': QualityAnalyzer._is_ultrasound_recording(stream_params.sample_rate or 44100)
        }
        
        # Analyze bitrate quality for lossy sources
        if codec_compression == AudioCompressionBaseType.LOSSY_COMPRESSED and stream_params.bit_rate:
            analysis['bitrate_class'] = QualityAnalyzer._classify_bitrate(
                stream_params.bit_rate, stream_params.nb_channels
            )
        
        # Determine overall quality tier
        analysis['quality_tier'] = QualityAnalyzer._determine_quality_tier(analysis)
        
        return analysis
    
    @staticmethod
    def _classify_bitrate(bitrate: int, channels: int) -> str:
        """
        Enhanced bitrate classification that accounts for inter-channel redundancy
        
        AAC uses joint stereo and parametric stereo techniques that exploit
        inter-channel redundancies, making simple per-channel division incorrect.
        """
        
        if channels == 1:  # Mono
            if bitrate < 86000:      # that can be critical for environment sound analysis
                return 'low'
            elif bitrate < 128000:
                return 'medium' 
            elif bitrate < 196000:
                return 'high'
            else:
                return 'excessive'
        
        elif channels == 2:  # Stereo - nicht einfach durch 2 teilen!
            # AAC Stereo-Effizienz: ~1.3-1.6x statt 2x Mono-Bitrate
            if bitrate < 110000:     
                return 'low'
            elif bitrate < 159000:   
                return 'medium'
            elif bitrate < 225000:  
                return 'high'
            else:
                return 'excessive'
        
        else:  # Multichannel (>2)
            # Für >2 Kanäle: AAC nutzt noch mehr Inter-Channel-Redundanzen
            per_channel_equivalent = bitrate / (channels * 0.7)  # 30% Effizienz-Bonus
            if per_channel_equivalent < 86000:
                return 'low'
            elif per_channel_equivalent < 128000:
                return 'medium'
            elif per_channel_equivalent < 196000:
                return 'high'
            else:
                return 'excessive'
    
    @staticmethod
    def _determine_quality_tier(analysis: dict) -> str:
        """Determine overall quality tier (low/standard/high/studio/ultrasound)"""
        compression = analysis['compression_type']
        sample_rate = analysis['sample_rate'] or 44100
        is_ultrasound = analysis.get('is_ultrasound', False)
        
        # Ultraschall bekommt eigene Kategorie
        if is_ultrasound:
            return 'ultrasound'
        
        if    compression == AudioCompressionBaseType.UNCOMPRESSED \
           or compression == AudioCompressionBaseType.LOSSLESS_COMPRESSED:
            if sample_rate >= 96000:
                return 'studio'
            elif sample_rate >= 48000:
                return 'high'
            else:
                return 'standard'
        else:  # LOSSY_COMPRESSED
            bitrate_class = analysis.get('bitrate_class', 'unknown')
            if bitrate_class in ['high', 'excessive']:
                return 'standard'  # Good lossy can be standard quality
            elif bitrate_class == 'medium':
                return 'standard'
            else:
                return 'low'
    
    @staticmethod
    def suggest_target_parameters(quality_analysis: dict) -> dict:
        """
        Suggest optimal target parameters based on source quality
        
        NEUE REGEL für Ultraschall:
        - Sample Rate > 96kS/s = Ultraschallaufnahme
        - Für Lossless: FLAC mit exakter Sample Rate (bis 655kHz FLAC-Limit)
        - Für Lossy: NIEMALS Resampling → nur Reinterpretation auf 32kS/s
        """
        compression = quality_analysis['compression_type']
        sample_rate = quality_analysis.get('sample_rate', 44100)
        
        suggestions = {}
        
        # ultrasonic is a special case 
        if QualityAnalyzer._is_ultrasound_recording(sample_rate):
            suggestions.update(QualityAnalyzer._suggest_ultrasound_parameters(quality_analysis))
            
        elif compression == AudioCompressionBaseType.LOSSY_COMPRESSED:
            # Standard-Lossy source → AAC with improved bitrate
            suggestions.update(QualityAnalyzer._suggest_aac_upgrade(quality_analysis))
            
        elif compression in [AudioCompressionBaseType.LOSSLESS_COMPRESSED, AudioCompressionBaseType.UNCOMPRESSED]:
            # Standard-Lossless source → FLAC to preserve quality
            suggestions.update(QualityAnalyzer._suggest_flac_preserve(quality_analysis))
            
        else:
            # Unknown compression → Conservative AAC choice
            suggestions.update(QualityAnalyzer._suggest_conservative_aac(quality_analysis))
        
        return suggestions
    
    @staticmethod
    def _suggest_ultrasound_parameters(quality_analysis: dict) -> dict:
        """
        Spezielle Behandlung für Ultraschallaufnahmen
        
        Regeln:
        - Lossless Quelle → FLAC mit exakter Sample Rate beibehalten
        - Lossy gewünscht → Reinterpretation auf 32kS/s (NIEMALS Resampling!)
        - Sample Rate > 655kHz → FLAC-Limit überschritten, spezielle Behandlung: Reinterpretation
        """
        compression = quality_analysis['compression_type']
        sample_rate = quality_analysis.get('sample_rate', 96000)
        
        # Für Lossless-Quellen: FLAC bevorzugen
        if compression in [AudioCompressionBaseType.LOSSLESS_COMPRESSED, AudioCompressionBaseType.UNCOMPRESSED]:
            
            if sample_rate <= 655350:  # Innerhalb FLAC-Grenzen
                return {
                    'target_format': TargetFormats.FLAC,  # Auto sample rate
                    'target_sampling_transform': TargetSamplingTransforming.EXACTLY,
                    'flac_compression_level': 6,  # Höhere Kompression für große Ultraschall-Dateien
                    'reason': f'Ultraschall-Aufnahme ({sample_rate//1000}kHz) → verlustfreie FLAC-Erhaltung'
                }
            else:
                # Über FLAC-Limit → spezielle Behandlung nötig
                return {
                    'target_format': TargetFormats.FLAC_96000,
                    'target_sampling_transform': TargetSamplingTransforming.REINTERPRETING_96000,
                    'flac_compression_level': 6,
                    'reason': f'Ultraschall {sample_rate//1000}kHz überschreitet FLAC-Limit → Re-Interpreting auf 96kHz'
                }
        
        # Für Lossy-Quellen oder wenn Lossy explizit gewünscht
        else:
            return {
                'target_format': TargetFormats.AAC_32000,
                'target_sampling_transform': TargetSamplingTransforming.REINTERPRETING_32000,
                'aac_bitrate': 192000,  # Höhere Bitrate für Ultraschall-Reinterpretation
                'reason': f'Ultraschall-Signal ({sample_rate//1000}kHz) → Reinterpretation auf 32kHz für AAC (16kHz Nyquist)'
            }
    
    @staticmethod
    def _suggest_aac_upgrade(quality_analysis: dict) -> dict:
        """Suggest AAC parameters for lossy source upgrade"""
        original_bitrate = quality_analysis.get('bit_rate', 128000)
        sample_rate = quality_analysis.get('sample_rate', 44100)
        
        # Increase bitrate by 10-40% to compensate for re-encoding loss
        if original_bitrate < 128000:
            target_bitrate = min(original_bitrate * 1.3, 192000)  # Significant boost for low quality
        else:
            target_bitrate = min(original_bitrate * 1.15, 320000)  # Moderate boost for good quality
        
        # Choose appropriate AAC format based on sample rate
        if sample_rate <= 48000:
            if sample_rate == 44100:
                target_format = TargetFormats.AAC_44100
            elif sample_rate == 48000:
                target_format = TargetFormats.AAC_48000
            else:
                target_format = TargetFormats.AAC  # Auto sample rate
        else:
            target_format = TargetFormats.AAC  # Let AAC handle high sample rates
        
        return {
            'target_format': target_format,
            'target_sampling_transform': TargetSamplingTransforming.EXACTLY,
            'aac_bitrate': int(target_bitrate),
            'reason': f'Lossy source upgrade: {original_bitrate//1000}kbps → {int(target_bitrate)//1000}kbps AAC'
        }
    
    @staticmethod
    def _suggest_flac_preserve(quality_analysis: dict) -> dict:
        """
        ANGEPASSTE METHODE: FLAC-Vorschläge mit Ultraschall-Awareness
        """
        sample_rate = quality_analysis.get('sample_rate', 44100)
        
        # Ultraschall-Bereich?
        if QualityAnalyzer._is_ultrasound_recording(sample_rate):
            # Delegiere an Ultraschall-Logik
            return QualityAnalyzer._suggest_ultrasound_parameters(quality_analysis)
        
        # Standard-Bereich: Bestehende Logik beibehalten
        if sample_rate == 44100:
            target_format = TargetFormats.FLAC_44100
        elif sample_rate == 48000:
            target_format = TargetFormats.FLAC_48000
        elif sample_rate == 88200:
            target_format = TargetFormats.FLAC_88200
        elif sample_rate == 96000:
            target_format = TargetFormats.FLAC_96000
        elif sample_rate == 176400:
            target_format = TargetFormats.FLAC_176400
        elif sample_rate == 192000:
            target_format = TargetFormats.FLAC_192000
        elif sample_rate <= 655350:  # Standard FLAC-Bereich
            target_format = TargetFormats.FLAC  # Auto sample rate
        else:
            # Sollte nicht erreicht werden, da Ultraschall-Check vorher greift
            target_format = TargetFormats.FLAC_192000
            return {
                'target_format': target_format,
                'target_sampling_transform': TargetSamplingTransforming.REINTERPRETING_96000,
                'reason': f'Sample rate {sample_rate//1000}kHz exceeds FLAC limit → reinterprete to 96kHz'
            }
        
        return {
            'target_format': target_format,
            'target_sampling_transform': TargetSamplingTransforming.EXACTLY,
            'flac_compression_level': 4,  # Balanced compression
            'reason': f'Lossless source → preserve with FLAC at {sample_rate//1000}kHz'
        }
    
    @staticmethod
    def _suggest_conservative_aac(quality_analysis: dict) -> dict:
        """Conservative AAC suggestion for unknown sources"""
        sample_rate = quality_analysis.get('sample_rate', 44100)
        channels = quality_analysis.get('channels', 2)
        
        # Conservative high-quality AAC
        bitrate = 192000 if channels == 2 else 160000
        
        if sample_rate == 44100:
            target_format = TargetFormats.AAC_44100
        elif sample_rate == 48000:
            target_format = TargetFormats.AAC_48000
        else:
            target_format = TargetFormats.AAC
        
        return {
            'target_format': target_format,
            'target_sampling_transform': TargetSamplingTransforming.EXACTLY,
            'aac_bitrate': bitrate,
            'reason': f'Unknown source quality → conservative {bitrate//1000}kbps AAC'
        }
#
# End of Class QualityAnalyzer
#
# ############################################################
# ############################################################

# ############################################################
# ############################################################
#
# Class ConflictAnalyzer  
# ======================
#
# Detects conflicts and quality issues
#
class ConflictAnalyzer:
    """Detects configuration conflicts and quality issues"""

    @staticmethod
    def analyze_conflicts(source_analysis: dict, target_params: dict) -> dict:
        """
        Analyze conflicts between source and target parameters
        
        Returns:
            dict: {
                'blocking_conflicts': [],     # Prevent import
                'quality_warnings': [],      # Quality degradation
                'efficiency_warnings': []    # Unnecessary bloat
            }
        """
        conflicts = {
            'blocking_conflicts': [],
            'quality_warnings': [],
            'efficiency_warnings': []
        }
        
        # Check target format compatibility
        target_format = target_params.get('target_format')
        if target_format:
            conflicts.update(ConflictAnalyzer._check_format_compatibility(
                source_analysis, target_format, target_params
            ))
        
        # Check sampling rate conflicts
        target_sampling = target_params.get('target_sampling_transform')
        if target_sampling:
            conflicts.update(ConflictAnalyzer._check_sampling_conflicts(
                source_analysis, target_sampling
            ))
        
        # Check quality degradation
        conflicts.update(ConflictAnalyzer._check_quality_degradation(
            source_analysis, target_params
        ))
        
        # Check efficiency issues (bloat)
        conflicts.update(ConflictAnalyzer._check_efficiency_issues(
            source_analysis, target_params
        ))
        
        return conflicts
    
    @staticmethod
    def _check_format_compatibility(source_analysis: dict, target_format: TargetFormats, target_params: dict) -> dict:
        """Check if target format is compatible with source"""
        conflicts = {'blocking_conflicts': [], 'quality_warnings': [], 'efficiency_warnings': []}
        
        source_rate = source_analysis.get('sample_rate', 44100)
        
        # Check if target format supports source sample rate
        if target_format.sample_rate and target_format.sample_rate != source_rate:
            # Sample rate mismatch - check if transform is specified
            transform = target_params.get('target_sampling_transform')
            if not transform or transform == TargetSamplingTransforming.EXACTLY:
                conflicts['blocking_conflicts'].append(
                    f"Sample rate mismatch: source {source_rate}Hz vs target {target_format.sample_rate}Hz. "
                    f"Specify target_sampling_transform for conversion."
                )
        
        return conflicts
    
    @staticmethod 
    def _check_sampling_conflicts(source_analysis: dict, target_sampling: TargetSamplingTransforming) -> dict:
        """Check sampling transformation conflicts"""
        conflicts = {'blocking_conflicts': [], 'quality_warnings': [], 'efficiency_warnings': []}
        
        source_rate = source_analysis.get('sample_rate', 44100)
        
        if target_sampling.sample_rate and target_sampling.code == "resampling":
            # Resampling quality check
            ratio = target_sampling.sample_rate / source_rate
            if ratio < 0.5:
                conflicts['quality_warnings'].append(
                    f"Significant downsampling: {source_rate}Hz → {target_sampling.sample_rate}Hz "
                    f"(ratio: {ratio:.2f}). Audio quality will be reduced."
                )
            elif ratio > 2.0:
                conflicts['efficiency_warnings'].append(
                    f"Significant upsampling: {source_rate}Hz → {target_sampling.sample_rate}Hz "
                    f"(ratio: {ratio:.2f}). File size will increase without quality benefit."
                )
        
        return conflicts
    
    @staticmethod
    def _check_quality_degradation(source_analysis: dict, target_params: dict) -> dict:
        """
        ERWEITERTE METHODE: Check for quality degradation scenarios + Ultraschall-Schutz
        """
        conflicts = {'blocking_conflicts': [], 'quality_warnings': [], 'efficiency_warnings': []}
        
        source_compression = source_analysis.get('compression_type')
        target_format = target_params.get('target_format')
        source_sample_rate = source_analysis.get('sample_rate', 44100)
        target_sampling = target_params.get('target_sampling_transform')
        is_ultrasound = source_analysis.get('is_ultrasound', False)
        
        # NEUE PRÜFUNG: Ultraschall-Schutz
        if is_ultrasound or source_sample_rate > QualityAnalyzer.ULTRASOUND_THRESHOLD:
            
            # Ultraschall + Lossy → Nur Reinterpretation erlaubt
            if target_format and target_format.code == 'aac':
                if target_sampling and target_sampling.code == 'resampling':
                    conflicts['blocking_conflicts'].append(
                        f"🚫 KRITISCH: Ultraschall-Aufnahme ({source_sample_rate//1000}kHz) + AAC + Resampling! "
                        f"Ultraschall-Signale werden zerstört. Nutze REINTERPRETING_32000 statt Resampling."
                    )
                elif not target_sampling or (target_sampling.code != 'reinterpreting' and target_sampling != TargetSamplingTransforming.EXACTLY):
                    conflicts['quality_warnings'].append(
                        f"⚠️  Ultraschall → AAC ohne Reinterpretation. "
                        f"Empfehlung: target_sampling_transform = 'REINTERPRETING_32000'"
                    )
            
            # Ultraschall + FLAC über Limit
            elif target_format and target_format.code == 'flac':
                if source_sample_rate > 655350:
                    conflicts['quality_warnings'].append(
                        f"⚠️  Ultraschall {source_sample_rate//1000}kHz überschreitet FLAC-Maximum (655kHz). "
                        f"Resampling auf 192kHz oder Reinterpretation + AAC erwägen."
                    )
        
        # Bestehende Qualitäts-Checks für Standard-Audio
        if (source_compression in [AudioCompressionBaseType.LOSSLESS_COMPRESSED, AudioCompressionBaseType.UNCOMPRESSED] 
            and target_format and target_format.code == 'aac'):
            
            # Lossless → Lossy conversion
            source_tier = source_analysis.get('quality_tier', 'unknown')
            target_bitrate = target_params.get('aac_bitrate', 160000)
            
            if source_tier == 'studio' and target_bitrate < 256000:
                conflicts['quality_warnings'].append(
                    f"Studio quality source → {target_bitrate//1000}kbps AAC. "
                    f"Consider higher bitrate (≥256kbps) or FLAC to preserve quality."
                )
            elif source_tier == 'high' and target_bitrate < 192000:
                conflicts['quality_warnings'].append(
                    f"High quality source → {target_bitrate//1000}kbps AAC. "
                    f"Consider ≥192kbps for better quality preservation."
                )
            elif source_tier == 'ultrasound':
                conflicts['quality_warnings'].append(
                    f"Ultraschall-Qualität → {target_bitrate//1000}kbps AAC. "
                    f"Stelle sicher, dass Reinterpretation auf 32kHz verwendet wird."
                )
        
        # Check for lossy → lossy re-encoding
        if (source_compression == AudioCompressionBaseType.LOSSY_COMPRESSED 
            and target_format and target_format.code == 'aac'):
            
            source_bitrate = source_analysis.get('bit_rate', 0)
            target_bitrate = target_params.get('aac_bitrate', 160000)
            
            if target_bitrate < source_bitrate * 1.2:  # Less than 20% increase
                conflicts['quality_warnings'].append(
                    f"Lossy re-encoding: {source_bitrate//1000}kbps → {target_bitrate//1000}kbps AAC. "
                    f"Quality loss expected. Consider ≥{int(source_bitrate*1.3)//1000}kbps."
                )
        
        return conflicts
    
    @staticmethod
    def _check_efficiency_issues(source_analysis: dict, target_params: dict) -> dict:
        """Check for unnecessary file size inflation"""
        conflicts = {'blocking_conflicts': [], 'quality_warnings': [], 'efficiency_warnings': []}
        
        source_compression = source_analysis.get('compression_type')
        target_format = target_params.get('target_format')
        
        # Check for unnecessary lossless encoding of low-quality sources
        if (source_compression == AudioCompressionBaseType.LOSSY_COMPRESSED 
            and target_format and target_format.code == 'flac'):
            
            source_bitrate_class = source_analysis.get('bitrate_class', 'unknown')
            if source_bitrate_class in ['low', 'medium']:
                conflicts['efficiency_warnings'].append(
                    f"Low/medium quality lossy source → FLAC. "
                    f"Consider AAC instead to avoid unnecessary file size inflation."
                )
        
        # Check for excessive AAC bitrates
        target_bitrate = target_params.get('aac_bitrate')
        if target_bitrate and target_bitrate > 256000:
            channels = source_analysis.get('channels', 2)
            per_channel = target_bitrate / channels
            if per_channel > 192000:
                conflicts['efficiency_warnings'].append(
                    f"Very high AAC bitrate: {target_bitrate//1000}kbps. "
                    f"Consider FLAC for lossless or lower AAC bitrate for efficiency."
                )
        
        return conflicts


# ############################################################
# ############################################################
#
# Class FileParameter (Enhanced)
# ==============================
#
# Analysis of the audio source file with intelligent suggestions
#

class FileParameter:
    """
    Enhanced file parameter analysis with intelligent suggestions and quality assessment
    """
    
    def __init__(self, file_path: str | pathlib.Path, user_meta: dict = {}, target_format: str = None, selected_audio_streams: int|list[int] = None):
        """
        Initialize FileParameter with optional initial target settings
        
        Args:
            file_path: Path to audio file
            user_meta: Additional user metadata
            target_format: Initial target format (optional)
            selected_audio_streams: Stream selection (optional)
        """
        logger.trace(f"Enhanced FileParameter initialization for: {str(file_path)}")
        
        # Core file analysis (unchanged from original)
        self._base_parameter = FileBaseParameters()
        
        file_path = pathlib.Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
            
        self._base_parameter.file = pathlib.Path(file_path).resolve()
        self._base_parameter.file_size_bytes = self._base_parameter.file.stat().st_size
           
        self._user_meta = user_meta if isinstance(user_meta, dict) else {}
        
        # NEW: Target parameter management
        self._user_defined_params: Set[str] = set()  # Track user-modified parameters
        self._target_format: TargetFormats|None = None
        self._target_sampling_transform: TargetSamplingTransforming|None = None
        self._aac_bitrate: int|None = None
        self._flac_compression_level: int = 4
        
        # Initialize core analysis data
        self._can_be_imported = False
        self._general_meta: dict = {}
        self._container: dict = {}
        self._audio_streams: List[dict] = []
        self._other_streams: List[dict] = []
        self._quality_analysis: dict = {}
        self._conflicts: dict = {}
        
        # Calculate file hash
        logger.trace("Calculating file hash...")
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        self._base_parameter.file_sh256 = hasher.hexdigest()
        
        # Extract file data
        logger.trace("Extracting file metadata...")
        self._extract_file_data()
        
        # Set initial parameters if provided
        if target_format:
            self.target_format = target_format
        if selected_audio_streams:
            self.audio_stream_selection_list = selected_audio_streams
        
        # Perform initial analysis
        logger.trace("Performing initial analysis...")
        self._analyze()
        
        logger.trace("Enhanced FileParameter initialization complete")
    
    # ================================================
    # TARGET PARAMETER PROPERTIES
    # ================================================
    
    @property
    def target_format(self) -> TargetFormats|None:
        """Current target format"""
        return self._target_format
    
    @target_format.setter
    def target_format(self, value: str|TargetFormats|None):
        """Set target format and trigger re-analysis"""
        if value is None:
            self._target_format = None
            self._user_defined_params.discard('target_format')
        else:
            self._target_format = TargetFormats.from_string_or_enum(value)
            self._user_defined_params.add('target_format')
        
        logger.trace(f"Target format set to: {self._target_format}")
        self._analyze()  # Auto-trigger re-analysis
    
    @property
    def target_sampling_transform(self) -> TargetSamplingTransforming|None:
        """Current target sampling transformation"""
        return self._target_sampling_transform
    
    @target_sampling_transform.setter  
    def target_sampling_transform(self, value: str|TargetSamplingTransforming|None):
        """Set target sampling transform and trigger re-analysis"""
        if value is None:
            self._target_sampling_transform = None
            self._user_defined_params.discard('target_sampling_transform')
        else:
            self._target_sampling_transform = TargetSamplingTransforming.from_string_or_enum(value)
            self._user_defined_params.add('target_sampling_transform')
        
        logger.trace(f"Target sampling transform set to: {self._target_sampling_transform}")
        self._analyze()
    
    @property
    def aac_bitrate(self) -> int|None:
        """Current AAC bitrate"""
        return self._aac_bitrate
    
    @aac_bitrate.setter
    def aac_bitrate(self, value: int|None):
        """Set AAC bitrate and trigger re-analysis"""
        if value is None:
            self._aac_bitrate = None
            self._user_defined_params.discard('aac_bitrate')
        else:
            if not isinstance(value, int) or value < 32000 or value > 320000:
                raise ValueError(f"AAC bitrate must be between 32000 and 320000, got: {value}")
            self._aac_bitrate = value
            self._user_defined_params.add('aac_bitrate')
        
        logger.trace(f"AAC bitrate set to: {self._aac_bitrate}")
        self._analyze()
    
    @property
    def flac_compression_level(self) -> int:
        """Current FLAC compression level"""
        return self._flac_compression_level
    
    @flac_compression_level.setter
    def flac_compression_level(self, value: int):
        """Set FLAC compression level and trigger re-analysis"""
        if not isinstance(value, int) or value < 0 or value > 12:
            raise ValueError(f"FLAC compression level must be between 0 and 12, got: {value}")
        self._flac_compression_level = value
        self._user_defined_params.add('flac_compression_level')
        
        logger.trace(f"FLAC compression level set to: {self._flac_compression_level}")
        self._analyze()
    
    # ================================================
    # ENHANCED ANALYSIS METHOD
    # ================================================
    
    def _analyze(self, target_format: str|None = None, audio_stream_selection_list: int|list[int]|None = None) -> bool:
        """
        Enhanced analysis with intelligent suggestions and conflict detection
        """
        logger.trace("Starting enhanced analysis...")
        
        # Handle parameter updates
        if target_format is not None:
            self.target_format = target_format
        if audio_stream_selection_list is not None:
            self.audio_stream_selection_list = audio_stream_selection_list
        
        # Perform base compatibility checks (from original implementation)
        self._can_be_imported = True
        
        # Validate selected streams exist
        if self._base_parameter.selected_audio_streams:
            known_indices = [sp.index_of_stream_in_file for sp in self._base_parameter.stream_parameters]
            for stream_index in self._base_parameter.selected_audio_streams:
                if stream_index not in known_indices:
                    self._can_be_imported = False
                    logger.error(f"Audio stream index {stream_index} not found in file")
                    return False
        else:
            # Auto-select first audio stream if none specified
            if self._base_parameter.stream_parameters:
                self._base_parameter.selected_audio_streams = [self._base_parameter.stream_parameters[0].index_of_stream_in_file]
        
        # Calculate total channels for import
        self._base_parameter.total_nb_of_channels_to_import = 0
        selected_indices = []
        for i, sp in enumerate(self._base_parameter.stream_parameters):
            if sp.index_of_stream_in_file in self._base_parameter.selected_audio_streams:
                selected_indices.append(i)
                self._base_parameter.total_nb_of_channels_to_import += sp.nb_channels or 0
        
        # Check stream compatibility
        if len(selected_indices) > 1:
            ref_stream:AudioStreamParameters = self._base_parameter.stream_parameters[selected_indices[0]]
            for i in selected_indices[1:]:
                stream:AudioStreamParameters = self._base_parameter.stream_parameters[i]
                if stream.nb_samples != ref_stream.nb_samples:
                    self._can_be_imported = False
                    logger.error("Different sample counts between streams")
                    return False
                if stream.sample_rate != ref_stream.sample_rate:
                    self._can_be_imported = False
                    logger.error("Different sample rates between streams")
                    return False
        
        # Check codec decodability
        for i in selected_indices:
            codec_name = self._base_parameter.stream_parameters[i].codec_name
            if not can_ffmpeg_decode_codec(codec_name):
                self._can_be_imported = False
                logger.error(f"Codec {codec_name} not decodable by ffmpeg")
                return False
        
        if not self._can_be_imported:
            return False
        
        # Perform quality analysis
        if selected_indices:
            primary_stream = self._base_parameter.stream_parameters[selected_indices[0]]
            self._quality_analysis = QualityAnalyzer.analyze_source_quality(
                primary_stream
            )
            
            # Apply intelligent suggestions for unset parameters
            self._apply_intelligent_suggestions()
            
            # Analyze conflicts and warnings
            self._analyze_conflicts()
        
        logger.trace(f"Enhanced analysis complete. Can import: {self._can_be_imported}")
        return self._can_be_imported
    
    def _apply_intelligent_suggestions(self):
        """Apply intelligent suggestions for parameters not set by user"""
        suggestions = QualityAnalyzer.suggest_target_parameters(self._quality_analysis)
        
        # Only apply suggestions for parameters not explicitly set by user
        if 'target_format' not in self._user_defined_params and 'target_format' in suggestions:
            self._target_format = suggestions['target_format']
            logger.trace(f"Auto-suggested target format: {self._target_format}")
        
        if 'target_sampling_transform' not in self._user_defined_params and 'target_sampling_transform' in suggestions:
            self._target_sampling_transform = suggestions['target_sampling_transform']
            logger.trace(f"Auto-suggested sampling transform: {self._target_sampling_transform}")
        
        if 'aac_bitrate' not in self._user_defined_params and 'aac_bitrate' in suggestions:
            self._aac_bitrate = suggestions['aac_bitrate']
            logger.trace(f"Auto-suggested AAC bitrate: {self._aac_bitrate}")
        
        if 'flac_compression_level' not in self._user_defined_params and 'flac_compression_level' in suggestions:
            self._flac_compression_level = suggestions['flac_compression_level']
            logger.trace(f"Auto-suggested FLAC compression: {self._flac_compression_level}")
    
    def _analyze_conflicts(self):
        """Analyze conflicts between source and target parameters"""
        target_params = {
            'target_format': self._target_format,
            'target_sampling_transform': self._target_sampling_transform,
            'aac_bitrate': self._aac_bitrate,
            'flac_compression_level': self._flac_compression_level
        }
        
        self._conflicts = ConflictAnalyzer.analyze_conflicts(self._quality_analysis, target_params)
        
        # Update can_be_imported based on blocking conflicts
        if self._conflicts['blocking_conflicts']:
            self._can_be_imported = False
            logger.warning(f"Import blocked by conflicts: {self._conflicts['blocking_conflicts']}")
        
        logger.trace(f"Conflict analysis complete: {len(self._conflicts['blocking_conflicts'])} blocking, "
                    f"{len(self._conflicts['quality_warnings'])} quality warnings, "
                    f"{len(self._conflicts['efficiency_warnings'])} efficiency warnings")

    # ================================================
    # ENHANCED PRINT OUTPUT (__str__ method)
    # ================================================
    
    def __str__(self) -> str:
        """
        Enhanced terminal output with quality analysis, suggestions, and warnings
        """
        return self._create_formatted_output()
    
    def _create_formatted_output(self) -> str:
        """Create beautifully formatted terminal output with dynamic box width (up to 180 chars)"""
        
        def get_display_width(text: str) -> int:
            """Get actual display width considering Unicode emoji/symbols that appear 2 chars wide"""
            # Known emoji/symbols that appear as 2 characters wide in terminals
            wide_chars = '🔧✅🟢📝💡⚠️🚫🦇'
            
            width = 0
            for char in text:
                if char in wide_chars:
                    width += 2  # These emoji appear 2 chars wide
                else:
                    width += 1  # Regular characters
            return width
        
        # Phase 1: Collect all content lines without box formatting
        content_lines = []
        
        # Constants for content limits
        MAX_TOTAL_LINE_LENGTH = 180
        CONFLICT_WRAP_LENGTH = 120
        MAX_CONTENT_LENGTH = MAX_TOTAL_LINE_LENGTH - 4  # Reserve space for "│ " and " │"
        
        def add_content_line(text: str, is_api_line: bool = False):
            """Add a content line, applying length limits based on type"""
            if is_api_line:
                # API lines can exceed normal limits and break out of box if needed
                content_lines.append(text)
            else:
                # Regular content gets truncated at max length (using display width)
                if get_display_width(text) > MAX_CONTENT_LENGTH:
                    # Truncate considering display width
                    truncated = ""
                    current_width = 0
                    for char in text:
                        char_width = 2 if char in '🔧✅🟢📝💡⚠️🚫🦇' else 1
                        if current_width + char_width + 3 > MAX_CONTENT_LENGTH:  # +3 for "..."
                            break
                        truncated += char
                        current_width += char_width
                    content_lines.append(truncated + "...")
                else:
                    content_lines.append(text)
        
        def add_header(title: str):
            """Add a section header (will be formatted with ├─ and ─┤ later)"""
            content_lines.append(f"HEADER:{title}")
        
        # Header with basic file info
        add_content_line("Audio File Analysis")
        add_content_line(f"File: {self._base_parameter.file.name}")
        
        file_size_mb = self._base_parameter.file_size_bytes / 1024 / 1024
        add_content_line(f"Size: {file_size_mb:.1f} MB, Container: {self._base_parameter.container_format_name}")
        add_content_line(f"SHA256: {self._base_parameter.file_sh256[:20]}...")
        # API access for basic file info
        add_content_line("📝 <your_instance>.base_parameter.file", is_api_line=True)
        add_content_line("📝 <your_instance>.base_parameter.file_size_bytes", is_api_line=True)
        add_content_line("📝 <your_instance>.base_parameter.file_sh256", is_api_line=True)
        
        # ================================================
        # CONTAINER METADATA SECTION
        # ================================================
        add_header("Container Metadata")
        
        container = self._container
        if container.get('format_long_name'):
            add_content_line(f"Format: {container['format_long_name']}")
        
        if container.get('duration'):
            duration = container['duration']
            duration_str = f"{int(duration//3600):02d}:{int((duration%3600)//60):02d}:{int(duration%60):02d}.{int((duration%1)*100):02d}"
            add_content_line(f"Duration: {duration_str}")
        
        if container.get('bit_rate'):
            total_bitrate = container['bit_rate'] // 1000
            add_content_line(f"Total Bitrate: {total_bitrate} kbps")
        
        if container.get('nb_streams'):
            add_content_line(f"Streams: {container['nb_streams']} total")
        
        # API access hints for known container metadata
        add_content_line("📝 <your_instance>.container['format_long_name']", is_api_line=True)
        add_content_line("📝 <your_instance>.container['duration']", is_api_line=True)
        add_content_line("📝 <your_instance>.container['bit_rate']", is_api_line=True)
        add_content_line("📝 <your_instance>.container['nb_streams']", is_api_line=True)
        
        # ================================================
        # ADDITIONAL CONTAINER METADATA (Unknown/Unexpected fields)
        # ================================================
        remaining_format = self._general_meta.get('format', {})
        if remaining_format:
            # Filter out tags (handled separately) and empty values
            additional_container = {k: v for k, v in remaining_format.items() 
                                if k != 'tags' and v is not None and str(v).strip()}
            
            if additional_container:
                add_header("Additional Container Metadata")
                
                # Show the additional fields
                for key, value in sorted(additional_container.items()):
                    add_content_line(f"{key}: {str(value)}")
                
                # API access hints - specific for each shown field
                for key in sorted(additional_container.keys()):
                    add_content_line(f"📝 <your_instance>.general_meta['format']['{key}']", is_api_line=True)
        
        # ================================================
        # AUDIO STREAMS DETAIL SECTION
        # ================================================
        add_header("Audio Streams Detail")
        
        for i, stream in enumerate(self._audio_streams):
            stream_marker = "→" if stream['index'] in self._base_parameter.selected_audio_streams else " "
            codec_name = stream['codec_name'] or 'unknown'
            add_content_line(f"{stream_marker}Stream #{stream['index']}: {codec_name}")
            
            # Basic stream info
            channels = stream.get('channels', 0)
            sample_rate = stream.get('sample_rate', 0)
            channel_text = f"{channels} channels" if channels != 2 else "stereo"
            if channels == 1:
                channel_text = "mono"
            
            # Ultraschall-Kennzeichnung für Streams
            ultrasound_marker = " 🦇" if sample_rate > QualityAnalyzer.ULTRASOUND_THRESHOLD else ""
            add_content_line(f"  {channel_text}, {sample_rate:,} Hz{ultrasound_marker}")
            
            # Channel layout
            if stream.get('channel_layout') and stream['channel_layout'] not in ['stereo', 'mono']:
                add_content_line(f"  Layout: {stream['channel_layout']}")
            
            # Sample format and bits per sample
            if stream.get('sample_fmt') or stream.get('bits_per_sample'):
                sample_info = f"{stream.get('sample_fmt', 'unknown')}"
                if stream.get('bits_per_sample'):
                    sample_info += f", {stream['bits_per_sample']} bits"
                add_content_line(f"  Sample: {sample_info}")
            
            # Stream-specific bitrate
            if stream.get('bit_rate'):
                stream_bitrate = stream['bit_rate'] // 1000
                add_content_line(f"  Bitrate: {stream_bitrate} kbps")
            
            # Disposition flags (if any interesting ones)
            if stream.get('disposition'):
                disp_flags = []
                for key, value in stream['disposition'].items():
                    if value and key in ['default', 'forced', 'comment', 'lyrics', 'karaoke']:
                        disp_flags.append(key)
                if disp_flags:
                    flags_str = ', '.join(disp_flags)
                    add_content_line(f"  Flags: {flags_str}")
            
            if i < len(self._audio_streams) - 1:  # Add separator between streams
                add_content_line("SEPARATOR")
        
        # API access hints for known audio stream fields
        add_content_line("📝 <your_instance>.audio_streams[0]['codec_name']", is_api_line=True)
        add_content_line("📝 <your_instance>.audio_streams[0]['channels']", is_api_line=True)
        add_content_line("📝 <your_instance>.audio_streams[0]['sample_rate']", is_api_line=True)
        add_content_line("📝 <your_instance>.audio_streams[0]['bit_rate']", is_api_line=True)
        
        # ================================================
        # ADDITIONAL AUDIO STREAM METADATA (Unknown/Unexpected fields)
        # ================================================
        general_streams = self._general_meta.get('streams', [])
        audio_general_streams = [s for s in general_streams if s.get('codec_type') == 'audio']
        
        if audio_general_streams:
            additional_audio_found = False
            
            for i, general_stream in enumerate(audio_general_streams):
                # Find any fields that are not empty and not already processed
                additional_fields = {k: v for k, v in general_stream.items() 
                                if v is not None and str(v).strip() and k != 'codec_type'}
                
                if additional_fields:
                    if not additional_audio_found:
                        add_header("Additional Audio Stream Metadata")
                        additional_audio_found = True
                    
                    stream_index = general_stream.get('index', i)
                    add_content_line(f"Stream #{stream_index} additional fields:")
                    
                    # Collect shown fields for API hints
                    shown_fields = []
                    shown_additional = 0
                    
                    for key, value in sorted(additional_fields.items()):
                        if shown_additional >= 6:  # Limit to 6 additional fields per stream
                            remaining_count = len(additional_fields) - shown_additional
                            add_content_line(f"  ... and {remaining_count} more fields")
                            break
                        
                        add_content_line(f"  {key}: {str(value)}")
                        shown_fields.append(key)
                        shown_additional += 1
                    
                    # Add specific API hints for shown fields
                    for field_key in shown_fields:
                        add_content_line(f"📝 <your_instance>.general_meta['streams'][{stream_index}]['{field_key}']", is_api_line=True)
        
        # ================================================
        # OTHER STREAMS SECTION (if any)
        # ================================================
        if self._other_streams:
            add_header("Other Streams")
            
            for stream in self._other_streams:
                codec_type = stream.get('codec_type', 'unknown')
                codec_name = stream.get('codec_name', 'unknown')
                add_content_line(f"Stream #{stream['index']}: {codec_type} ({codec_name})")
            
            # API access hints
            add_content_line("📝 <your_instance>.other_streams[0]['codec_type']", is_api_line=True)
            add_content_line("📝 <your_instance>.other_streams[0]['codec_name']", is_api_line=True)
        
        # ================================================
        # METADATA TAGS SECTION (Complete, not just important ones)
        # ================================================
        format_tags = self._general_meta.get('format', {}).get('tags', {})
        if format_tags:
            add_header("Metadata Tags")
            
            # Show ALL tags, but prioritize important ones first
            important_tags = ['title', 'artist', 'album', 'date', 'genre', 'comment']
            displayed_tags = []
            
            # First show important tags
            for tag in important_tags:
                if tag in format_tags:
                    add_content_line(f"{tag.capitalize()}: {str(format_tags[tag])}")
                    displayed_tags.append(tag)
            
            # Then show all remaining tags
            remaining_tags = {k: v for k, v in format_tags.items() if k not in displayed_tags}
            if remaining_tags:
                if displayed_tags:  # Add separator if we showed important tags first
                    add_content_line("--- Additional Tags ---")
                
                for tag, value in sorted(remaining_tags.items()):
                    add_content_line(f"{tag}: {str(value)}")
                    displayed_tags.append(tag)
            
            # API access hints for all displayed tags
            if len(displayed_tags) <= 3:
                # Show individual access for few tags
                for tag in displayed_tags:
                    add_content_line(f"📝 <your_instance>.general_meta['format']['tags']['{tag}']", is_api_line=True)
            else:
                # Show individual access for each tag (no more templates or examples)
                for tag in displayed_tags:
                    add_content_line(f"📝 <your_instance>.general_meta['format']['tags']['{tag}']", is_api_line=True)
                # Add general access pattern as additional info
                add_content_line("📝 <your_instance>.general_meta['format']['tags']  # All tags dict", is_api_line=True)
        
        # ================================================
        # IMPORT SETTINGS - AM ENDE! (Most important for user)
        # ================================================
        add_header("Recommended Import Settings")
        
        # Primary audio stream info for import
        if self._base_parameter.stream_parameters:
            primary_stream = self._base_parameter.stream_parameters[0]
            codec_name = primary_stream.codec_name or "unknown"
            compression_type = self._quality_analysis.get('compression_type', AudioCompressionBaseType.UNKNOWN)
            
            # Format compression type display
            compression_display = {
                AudioCompressionBaseType.UNCOMPRESSED: "uncompressed",
                AudioCompressionBaseType.LOSSLESS_COMPRESSED: "lossless",
                AudioCompressionBaseType.LOSSY_COMPRESSED: "lossy",
                AudioCompressionBaseType.UNKNOWN: "unknown"
            }.get(compression_type, "unknown")
            
            add_content_line(f"Source: {codec_name} ({compression_display})")
        
        # Show current/suggested parameters
        if self._target_format:
            status_icon = "✅" if 'target_format' in self._user_defined_params else "🔧"
            add_content_line(f"{status_icon} Target: {self._target_format.name}")
        
        if self._target_sampling_transform:
            status_icon = "✅" if 'target_sampling_transform' in self._user_defined_params else "🔧"
            add_content_line(f"{status_icon} Sampling: {self._target_sampling_transform.name}")
        
        # Codec-specific parameters
        if self._target_format and self._target_format.code == 'aac' and self._aac_bitrate:
            status_icon = "✅" if 'aac_bitrate' in self._user_defined_params else "🔧"
            add_content_line(f"{status_icon} Bitrate: AAC {self._aac_bitrate//1000} kbps")
        
        if self._target_format and self._target_format.code == 'flac':
            status_icon = "✅" if 'flac_compression_level' in self._user_defined_params else "🔧"
            add_content_line(f"{status_icon} Compression: FLAC level {self._flac_compression_level}")
        
        # Show reasoning if suggestions were applied
        suggestions = QualityAnalyzer.suggest_target_parameters(self._quality_analysis)
        if 'reason' in suggestions and not all(param in self._user_defined_params for param in ['target_format', 'aac_bitrate', 'flac_compression_level']):
            # Simple highlighted rationale line instead of complex box
            reason = suggestions['reason']
            add_content_line(f">>> {reason} <<<")
        
        # API access hints for import parameters - conditional based on target format
        add_content_line("📝 <your_instance>.target_format", is_api_line=True)
        add_content_line("📝 <your_instance>.target_sampling_transform", is_api_line=True)
        
        # Codec-specific API hints - only show relevant ones
        if self._target_format and self._target_format.code == 'aac':
            add_content_line("📝 <your_instance>.aac_bitrate", is_api_line=True)
        elif self._target_format and self._target_format.code == 'flac':
            add_content_line("📝 <your_instance>.flac_compression_level", is_api_line=True)
        else:
            # If no specific target format, show both with note
            add_content_line("📝 <your_instance>.aac_bitrate  # if using AAC", is_api_line=True)
            add_content_line("📝 <your_instance>.flac_compression_level  # if using FLAC", is_api_line=True)
        
        add_content_line("📝 <your_instance>.get_import_parameters()", is_api_line=True)
        
        # Warnings and conflicts section
        if self._conflicts:
            if self._conflicts['quality_warnings'] or self._conflicts['efficiency_warnings'] or self._conflicts['blocking_conflicts']:
                add_header("Warnings & Issues")
                
                # Blocking conflicts (critical)
                for conflict in self._conflicts['blocking_conflicts']:
                    conflict_text = f"🚫 {conflict}"
                    wrapped_lines = self._wrap_text(conflict_text, CONFLICT_WRAP_LENGTH)
                    for line in wrapped_lines:
                        add_content_line(line)
                
                # Quality warnings
                for warning in self._conflicts['quality_warnings']:
                    warning_text = f"⚠️  {warning}"
                    wrapped_lines = self._wrap_text(warning_text, CONFLICT_WRAP_LENGTH)
                    for line in wrapped_lines:
                        add_content_line(line)
                
                # Efficiency warnings
                for warning in self._conflicts['efficiency_warnings']:
                    warning_text = f"💡 {warning}"
                    wrapped_lines = self._wrap_text(warning_text, CONFLICT_WRAP_LENGTH)
                    for line in wrapped_lines:
                        add_content_line(line)
                
                # API access for conflicts
                add_content_line("📝 <your_instance>.conflicts", is_api_line=True)
                add_content_line("📝 <your_instance>.has_blocking_conflicts", is_api_line=True)
        
        # Status section - FINAL
        add_header("Status")
        if self._can_be_imported:
            add_content_line("🟢 Ready for import")
        else:
            add_content_line("🔴 Import blocked - resolve conflicts above")
        
        # API access for status
        add_content_line("📝 <your_instance>.can_be_imported", is_api_line=True)
        
        # ================================================
        # OTHER FILE METADATA (chapters, programs, etc.)
        # ================================================
        other_metadata_found = False
        
        # Check for chapters
        chapters = self._general_meta.get('chapters', [])
        if chapters:
            if not other_metadata_found:
                add_header("Other File Metadata")
                other_metadata_found = True
            
            add_content_line(f"Chapters: {len(chapters)} found")
            # Show first few chapters as examples
            for i, chapter in enumerate(chapters[:2]):  # Limit to first 2 chapters
                title = chapter.get('tags', {}).get('title', f'Chapter {i+1}')
                start_time = chapter.get('start_time', 'unknown')
                end_time = chapter.get('end_time', 'unknown')
                add_content_line(f"  {title}: {start_time}s - {end_time}s")
            
            if len(chapters) > 2:
                add_content_line(f"  ... and {len(chapters)-2} more chapters")
            
            add_content_line("📝 <your_instance>.general_meta['chapters']", is_api_line=True)
            add_content_line("📝 Example: ...['chapters'][0]['tags']['title']", is_api_line=True)
        
        # Check for programs
        programs = self._general_meta.get('programs', [])
        if programs:
            if not other_metadata_found:
                add_header("Other File Metadata")
                other_metadata_found = True
            
            add_content_line(f"Programs: {len(programs)} found")
            add_content_line("📝 <your_instance>.general_meta['programs']", is_api_line=True)
        
        # Check for any other top-level keys we haven't processed
        processed_top_level = {'format', 'streams', 'chapters', 'programs'}
        remaining_top_level = {k: v for k, v in self._general_meta.items() 
                            if k not in processed_top_level and k != 'user_meta' 
                            and v is not None and str(v).strip()}
        
        if remaining_top_level:
            if not other_metadata_found:
                add_header("Other File Metadata")
                other_metadata_found = True
            
            for key, value in sorted(remaining_top_level.items()):
                # Handle different types of values
                if isinstance(value, (dict, list)):
                    count = len(value) if hasattr(value, '__len__') else 'complex'
                    add_content_line(f"{key}: {count} items")
                else:
                    add_content_line(f"{key}: {str(value)}")
                
                # Specific API access for each field
                add_content_line(f"📝 <your_instance>.general_meta['{key}']", is_api_line=True)
        
        # Legend for icons - FINAL
        add_content_line("")  # Empty line
        legend_text = "Legend: ✅=User set, 🔧=Auto-suggested, 📝=API access"
        if self._quality_analysis.get('is_ultrasound', False):
            legend_text += ", 🦇=Ultrasound"
        
        # Split legend if too long (using CONFLICT_WRAP_LENGTH)
        legend_lines = self._wrap_text(legend_text, CONFLICT_WRAP_LENGTH)
        for legend_line in legend_lines:
            add_content_line(legend_line)
        
        # ================================================
        # Phase 2: Format all content with dynamic box width using display width
        # ================================================
        
        # Find the maximum content display width (exclude rationale box elements from calculation)
        max_content_width = 0
        for line in content_lines:
            # Skip special markers and rationale elements when calculating max width
            if line.startswith(('HEADER:', 'SEPARATOR', 'RATIONALE_')):
                continue
            max_content_width = max(max_content_width, get_display_width(line))
        
        # Calculate box width (content + "│ " + " │")
        box_width = max_content_width + 4
        
        # Format all lines with proper box characters using display width for padding
        formatted_lines = []
        
        # Top border
        formatted_lines.append("╭" + "─" * (box_width - 2) + "╮")
        
        in_rationale = False
        
        for line in content_lines:
            if line.startswith("HEADER:"):
                # Format header with full border - must match exact box width
                title = line[7:]  # Remove "HEADER:" prefix
                title_width = get_display_width(title)
                
                # Total available space for the header line content (excluding ├─ and ┤)
                # box_width includes the outer borders, so available space is box_width - 4
                available_space = box_width - 4  # -4 for "├─" at start and "┤" at end
                
                # Space needed: "─ " (2) + title + " " + remaining "─" characters
                title_and_spaces = 2 + title_width  # "─ " + title
                remaining_dashes = available_space - title_and_spaces + 1  # +1 for perfect alignment!
                
                if remaining_dashes < 0:
                    # Title too long, truncate
                    max_title_space = available_space - 2  # Reserve space for "─ "
                    truncated = ""
                    current_width = 0
                    for char in title:
                        char_width = 2 if char in '🔧✅🟢📝💡⚠️🚫🦇' else 1
                        if current_width + char_width + 3 > max_title_space:  # +3 for "..."
                            break
                        truncated += char
                        current_width += char_width
                    title = truncated + "..."
                    title_width = get_display_width(title)
                    title_and_spaces = 2 + title_width
                    remaining_dashes = available_space - title_and_spaces + 1  # +1 for perfect alignment!
                    if remaining_dashes < 0:
                        remaining_dashes = 0
                
                header_line = f"├─ {title} {'─' * remaining_dashes}┤"
                formatted_lines.append(header_line)
                
            elif line == "SEPARATOR":
                # Add separator line
                formatted_lines.append("│" + "─" * (box_width - 2) + "│")
                
            else:
                # Regular content line (removed all rationale box handling)
                line_width = get_display_width(line)
                if line_width > MAX_TOTAL_LINE_LENGTH:
                    # API line that exceeds limit - break out of box
                    formatted_lines.append(line)
                else:
                    # Normal line with padding based on display width
                    padding = max_content_width - line_width
                    if padding < 0:
                        padding = 0
                    formatted_lines.append(f"│ {line}{' ' * padding} │")
        
        # Bottom border
        formatted_lines.append("╰" + "─" * (box_width - 2) + "╯")
        
        return "\n".join(formatted_lines)



    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to specified width, preserving words"""
        import textwrap
        return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    
    # ================================================
    # EXISTING METHODS (kept for compatibility)
    # ================================================
    
    def _validate_and_save_user_meta(self, user_meta: dict|None):
        """Validate and save user metadata"""
        if not isinstance(user_meta, (dict, type(None))):
            raise ValueError("Additional information, given as user_meta, must be structured as dictionary.")
        if user_meta is not None:
            logger.trace(f"user_meta value accepted: {user_meta}")
            self._user_meta = user_meta
            return True
        return False

    def _validate_and_save_selected_audio_streams(self, audio_stream_selection_list: int|list[int]|None):
        """Validate and save selected audio streams"""
        if audio_stream_selection_list is None:
            return False
        if isinstance(audio_stream_selection_list, int):
            audio_stream_selection_list = [audio_stream_selection_list]
        if not isinstance(audio_stream_selection_list, list):
            raise ValueError("audio_stream_selection_list must be int or list[int].")
        if not all(isinstance(i, int) for i in audio_stream_selection_list):
            raise ValueError("audio_stream_selection_list: all elements must be integers.")
        if len(audio_stream_selection_list) < 1:
            raise ValueError("At least one audio stream selection needed.")
        
        # TODO for next versions: Accept and process more than one stream
        if len(audio_stream_selection_list) > 1:
            raise ValueError("More than exactly one selected audio stream not yet supported. But, planned for next versions.")
        
        self._base_parameter.selected_audio_streams = audio_stream_selection_list
        return True
    
    def _extract_file_data(self):
        """Extract complete file metadata using ffprobe"""
        logger.trace(f"Extracting metadata for file: {self._base_parameter.file.name}")
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_format",
                "-show_streams", 
                "-show_chapters",
                "-show_programs",
                "-of", "json",
                str(self._base_parameter.file)
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                timeout=30
            )
            
            ffprobe_data = json.loads(result.stdout)
            
            self._extract_container_info(ffprobe_data, self._base_parameter.file.name)
            self._extract_stream_info(ffprobe_data, self._base_parameter.file.name)
            self._extract_general_meta(ffprobe_data, self._base_parameter.file.name)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"ffprobe failed with return code {e.returncode}")
            raise RuntimeError(f"Failed to analyze media file {self._base_parameter.file.name}") from e
        except subprocess.TimeoutExpired as e:
            logger.error(f"ffprobe timed out after 30 seconds")
            raise RuntimeError(f"Media file analysis timed out: {self._base_parameter.file.name}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON output from ffprobe: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error during ffprobe analysis: {e}")
    
    def _extract_container_info(self, ffprobe_data: dict, file: str):
        """Extract container-level information"""
        logger.trace(f"Extracting container information for file '{file}'")
        format_info = ffprobe_data.get("format", {})
        
        self._base_parameter.container_format_name = format_info.get("format_name")
        self._container = {
            "format_name": format_info.get("format_name"),
            "format_long_name": format_info.get("format_long_name"),
            "duration": safe_float_conversion(format_info.get("duration")),
            "size": safe_int_conversion(format_info.get("size")),
            "bit_rate": safe_int_conversion(format_info.get("bit_rate")),
            "probe_score": safe_int_conversion(format_info.get("probe_score")),
            "start_time": safe_float_conversion(format_info.get("start_time")),
            "nb_streams": safe_int_conversion(format_info.get("nb_streams")),
            "nb_programs": safe_int_conversion(format_info.get("nb_programs"))
        }
        logger.trace(f"Container info extracted for file '{file}'")
    
    def _extract_stream_info(self, ffprobe_data: dict, file: str):
        """Extract stream-level information"""
        all_streams = ffprobe_data.get("streams", [])
        
        self._audio_streams = []
        self._other_streams = []
        
        for stream_data in all_streams:
            if stream_data.get("codec_type") == "audio":
                self._create_audio_stream_info_dict(stream_data, file)
            else:
                self._create_other_stream_info_dict(stream_data, file)
                
        # Sort by index
        self._audio_streams.sort(key=lambda x: x["index"] if x["index"] is not None else 999)
        self._other_streams.sort(key=lambda x: x["index"] if x["index"] is not None else 999)
    
    def _create_audio_stream_info_dict(self, stream_data: dict, file: str):
        """Create audio stream info dictionary"""
        logger.trace(f"Extracting audio stream information of file '{file}'")
        
        # Extract bitrate for quality analysis
        bit_rate = safe_int_conversion(stream_data.get("bit_rate"))
        
        self._audio_streams.append({
            "index": safe_int_conversion(stream_data.get("index")),
            "id": stream_data.get("id"),
            "codec_name": stream_data.get("codec_name"),
            "codec_long_name": stream_data.get("codec_long_name"),
            "codec_type": stream_data.get("codec_type"),
            "codec_tag": stream_data.get("codec_tag"),
            "codec_tag_string": stream_data.get("codec_tag_string"),
            "sample_rate": safe_int_conversion(stream_data.get("sample_rate")),
            "sample_fmt": stream_data.get("sample_fmt"),
            "channels": safe_int_conversion(stream_data.get("channels")),
            "channel_layout": stream_data.get("channel_layout"),
            "bits_per_sample": safe_int_conversion(stream_data.get("bits_per_sample")),
            "bit_rate": bit_rate,
            "duration": safe_float_conversion(stream_data.get("duration")),
            "start_time": safe_float_conversion(stream_data.get("start_time")),
            "time_base": stream_data.get("time_base"),
            "start_pts": safe_int_conversion(stream_data.get("start_pts")),
            "duration_ts": safe_int_conversion(stream_data.get("duration_ts")),
            "disposition": stream_data.get("disposition", {})
        })
        
        # Enhanced AudioStreamParameters with bitrate
        self._base_parameter.stream_parameters.append(
            AudioStreamParameters(
                index_of_stream_in_file=safe_int_conversion(stream_data.get("index")),
                nb_channels=safe_int_conversion(stream_data.get("channels")),
                sample_rate=safe_int_conversion(stream_data.get("sample_rate")),
                sample_format=stream_data.get("sample_fmt"),
                codec_name=stream_data.get("codec_name"),
                nb_samples=self._calculate_total_samples(self._audio_streams[-1]),
                bit_rate=bit_rate  # NEW: Include bitrate
            )
        )
        self._base_parameter.total_nb_of_channels += safe_int_conversion(stream_data.get("channels"))

    def _create_other_stream_info_dict(self, stream_data: dict, file: str):
        """Create non-audio stream info dictionary"""
        logger.trace(f"Extracting non-audio stream information of file '{file}'")
        self._other_streams.append({
            "index": safe_int_conversion(stream_data.get("index")),
            "id": stream_data.get("id"),
            "codec_name": stream_data.get("codec_name"),
            "codec_long_name": stream_data.get("codec_long_name"),
            "codec_type": stream_data.get("codec_type"),
            "codec_tag": stream_data.get("codec_tag"),
            "codec_tag_string": stream_data.get("codec_tag_string"),
            "duration": safe_float_conversion(stream_data.get("duration")),
            "start_time": safe_float_conversion(stream_data.get("start_time")),
            "time_base": stream_data.get("time_base"),
            "start_pts": safe_int_conversion(stream_data.get("start_pts")),
            "duration_ts": safe_int_conversion(stream_data.get("duration_ts")),
            "disposition": stream_data.get("disposition", {})
        })
    
    def _extract_general_meta(self, ffprobe_data: dict, file: str):
        """Extract general metadata"""
        import copy
        logger.trace(f"Extracting general metadata of file '{file}'")
        
        self._general_meta = copy.deepcopy(ffprobe_data)
        self._remove_processed_container_paths(self._general_meta)
        self._remove_processed_stream_paths(self._general_meta)
        self._general_meta["user_meta"] = self._user_meta
    
    def _remove_processed_container_paths(self, meta_data: dict):
        """Remove already processed container paths"""
        format_section = meta_data.get("format", {})
        processed_container_keys = [
            "format_name", "format_long_name", "duration", "size", 
            "bit_rate", "probe_score", "start_time", "nb_streams", "nb_programs"
        ]
        for key in processed_container_keys:
            format_section.pop(key, None)
    
    def _remove_processed_stream_paths(self, meta_data: dict):
        """Remove already processed stream paths"""
        streams_section = meta_data.get("streams", [])
        processed_audio_keys = [
            "index", "id", "codec_name", "codec_long_name", "codec_type",
            "codec_tag", "codec_tag_string", "sample_rate", "sample_fmt",
            "channels", "channel_layout", "bits_per_sample", "bit_rate",
            "duration", "start_time", "time_base", "start_pts", "duration_ts",
            "disposition"
        ]
        processed_other_keys = [
            "index", "id", "codec_name", "codec_long_name", "codec_type",
            "codec_tag", "codec_tag_string", "duration",
            "start_time", "time_base", "start_pts", "duration_ts", "disposition"
        ]
        
        for stream in streams_section:
            codec_type = stream.get("codec_type")
            if codec_type == "audio":
                for key in processed_audio_keys:
                    stream.pop(key, None)
            else:
                for key in processed_other_keys:
                    stream.pop(key, None)
     
    def _calculate_total_samples(self, stream: dict) -> Optional[int]:
        """Calculate total samples in audio stream"""
        # Method 1: duration_ts (best - direct sample count)
        duration_ts = stream.get("duration_ts")
        time_base = stream.get("time_base")
        sample_rate = stream.get("sample_rate")
        
        if duration_ts and time_base and sample_rate:
            try:
                if '/' in str(time_base):
                    num, den = map(int, str(time_base).split('/'))
                    if den == sample_rate and num == 1:
                        return duration_ts
            except (ValueError, ZeroDivisionError):
                pass
        
        # Method 2: duration * sample_rate (standard)
        duration = stream.get("duration")
        if duration and sample_rate:
            return int(duration * sample_rate)
        
        # Method 3: Container duration fallback
        container_duration = self._container.get("duration")
        if container_duration and sample_rate:
            return int(container_duration * sample_rate)
        
        return None
    
    # ================================================
    # PUBLIC API PROPERTIES (Enhanced)
    # ================================================
    
    @property
    def base_parameter(self) -> FileBaseParameters:
        """Base parameters for import"""
        return self._base_parameter
    
    @property
    def container(self) -> dict:
        """Container level information"""
        return self._container.copy()
    
    @property
    def audio_streams(self) -> List[dict]:
        """List of parameters of audio streams"""
        return [stream.copy() for stream in self._audio_streams]
    
    @property
    def selected_audio_streams(self) -> List[AudioStreamParameters]:
        """List of parameters of pre-selected audio streams"""
        if len(self._base_parameter.selected_audio_streams) < 1:
            return []
        return [stream_parameter for stream_parameter in self._base_parameter.stream_parameters 
                if stream_parameter.index_of_stream_in_file in self._base_parameter.selected_audio_streams]
    
    @property
    def other_streams(self) -> List[dict]:
        """List of parameters of non-audio streams"""
        return [stream.copy() for stream in self._other_streams]
    
    @property
    def general_meta(self) -> dict:
        """General meta data (Tags, etc.)"""
        return self._general_meta.copy()
    
    @property
    def user_meta(self) -> dict:
        """User defined meta data"""
        return self._user_meta.copy()
    
    @user_meta.setter
    def user_meta(self, user_meta: dict):
        """Set user meta data"""
        self._validate_and_save_user_meta(user_meta)
        logger.trace("user_meta data set.")
       
    @property
    def audio_stream_selection_list(self) -> list[int]:
        """Selected audio streams for import"""
        return self._base_parameter.selected_audio_streams.copy()
    
    @audio_stream_selection_list.setter
    def audio_stream_selection_list(self, audio_stream_selection_list: list[int]):
        """Set selected audio streams and trigger re-analysis"""
        self._validate_and_save_selected_audio_streams(audio_stream_selection_list)
        logger.trace("audio_stream_selection_list set. Re-running analysis.")
        self._analyze()
    
    @property
    def can_be_imported(self) -> bool:
        """Flag if the import is permitted"""
        return self._can_be_imported

    @property
    def has_audio(self) -> bool:
        """Check if audio streams are present"""
        return len(self._audio_streams) > 0
    
    @property
    def number_of_audio_streams(self) -> int:
        """Number of audio streams"""
        return len(self._audio_streams)
    
    # NEW: Quality and conflict properties
    @property
    def quality_analysis(self) -> dict:
        """Quality analysis results"""
        return self._quality_analysis.copy()
    
    @property
    def conflicts(self) -> dict:
        """Conflict analysis results"""
        return self._conflicts.copy()
    
    @property
    def has_blocking_conflicts(self) -> bool:
        """Check if there are blocking conflicts"""
        return bool(self._conflicts.get('blocking_conflicts', []))
    
    @property
    def has_quality_warnings(self) -> bool:
        """Check if there are quality warnings"""
        return bool(self._conflicts.get('quality_warnings', []))
    
    @property
    def has_efficiency_warnings(self) -> bool:
        """Check if there are efficiency warnings"""
        return bool(self._conflicts.get('efficiency_warnings', []))
    
    @property
    def is_ultrasound_recording(self) -> bool:
        """Check if this is an ultrasound recording (>96kHz)"""
        return self._quality_analysis.get('is_ultrasound', False)
    
    # ================================================
    # PUBLIC API METHODS
    # ================================================
    
    def analyze(self, target_format: str|None = None, audio_stream_selection_list: int|list[int]|None = None) -> bool:
        """Public method to trigger analysis"""
        return self._analyze(target_format, audio_stream_selection_list)
    
    def reset_suggestions(self):
        """Reset all auto-suggested parameters to allow re-suggestion"""
        self._user_defined_params.clear()
        self._analyze()
        logger.trace("All suggestions reset, re-analysis triggered")
    
    def get_import_parameters(self) -> dict:
        """Get all parameters needed for import process"""
        return {
            'target_format': self._target_format,
            'target_sampling_transform': self._target_sampling_transform,
            'aac_bitrate': self._aac_bitrate,
            'flac_compression_level': self._flac_compression_level,
            'selected_streams': self._base_parameter.selected_audio_streams.copy(),
            'file_path': self._base_parameter.file,
            'user_meta': self._user_meta.copy()
        }


# End of Class FileParameter
#    
# ############################################################
# ############################################################

logger.trace("Enhanced FileParameter module loaded.")