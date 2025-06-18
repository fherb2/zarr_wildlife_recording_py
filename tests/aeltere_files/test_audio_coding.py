import pytest
import subprocess
import re
from typing import Set
from zarrwlr.audio_coding import AudioCompressionBaseType, audio_codec_compression

def get_ffmpeg_audio_decoders() -> Set[str]:
    """
    Holt alle Audio-Decoder von ffmpeg über die Kommandozeile.
    
    Returns:
        Set[str]: Menge aller Audio-Decoder-Namen, die ffmpeg unterstützt
    """
    try:
        # ffmpeg -decoders gibt alle verfügbaren Decoder aus
        result = subprocess.run(
            ["ffmpeg", "-decoders"],
            capture_output=True,
            text=True,
            check=True
        )
        
        audio_decoders = set()
        lines = result.stdout.split('\n')
        
        print("DEBUG: ffmpeg -decoders output analysis:")
        print(f"Total lines: {len(lines)}")
        
        # Debug: Zeige die ersten 20 Zeilen
        print("\nFirst 20 lines:")
        for i, line in enumerate(lines[:20]):
            print(f"{i:2d}: '{line}'")
        print("\n----- first 20 Lines ---------\n\n")
        
        # Robustere Parsing-Strategie
        decoder_section_started = False
        
        for i, line in enumerate(lines):
            # Decoder-Sektion beginnt nach einer Linie mit Bindestrichen
            if "------" in line:
                decoder_section_started = True
                print(f"Decoder section started at line {i}")
                continue
                
            if not decoder_section_started:
                continue
                
            # Leerzeilen überspringen
            if not line.strip():
                continue
            
            # Verschiedene Parsing-Ansätze probieren
            # Format kann variieren: " A..... codec_name    Description" oder " A..... codec_name"
            
            # Ansatz 1: Standard-Pattern
            match1 = re.match(r'^\s*([VAS][FL\.]{5})\s+(\S+)', line)
            # Ansatz 2: Flexiblerer Pattern
            match2 = re.match(r'^\s*([VAS][FL\.]*)\s+(\S+)', line)
            # Ansatz 3: Noch flexibler
            match3 = re.match(r'^\s*([VAS]\S*)\s+(\S+)', line)
            
            match = match1 or match2 or match3
            
            if match:
                flags, codec_name = match.groups()
                # Audio-Decoder erkennen (beginnen mit 'A')
                if flags.startswith('A'):
                    audio_decoders.add(codec_name)
                    print(f"  -> Added audio codec: {codec_name}")
            elif line.strip() and decoder_section_started and i < 50:
                # Debug: Zeige nicht-gematchte Zeilen in der Decoder-Sektion
                print(f"  No match for: '{line.strip()}'")
        
        print(f"\nTotal audio decoders found: {len(audio_decoders)}")
        return audio_decoders
        
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg command failed: {e}")
        return set()
    except FileNotFoundError:
        print("ffmpeg not found in PATH")
        return set()


class TestAudioCodecCompression:
    """Test-Klasse für die audio_codec_compression Funktion"""
    
    @pytest.fixture(scope="class")
    def ffmpeg_audio_codecs(self) -> Set[str]:
        """Fixture, die alle ffmpeg Audio-Decoder lädt"""
        return get_ffmpeg_audio_decoders()
    
    def test_ffmpeg_codecs_coverage(self, ffmpeg_audio_codecs: Set[str]):
        """
        Testet, ob alle ffmpeg Audio-Decoder von der Funktion erkannt werden.
        Gibt unerkannte Codecs aus und überprüft die Abdeckung.
        """
        unknown_codecs = []
        recognized_codecs = {
            AudioCompressionBaseType.UNCOMPRESSED: [],
            AudioCompressionBaseType.LOSSLESS_COMPRESSED: [],
            AudioCompressionBaseType.LOSSY_COMPRESSED: [],
            AudioCompressionBaseType.UNKNOWN: []
        }
        
        print("\n=== FFMPEG AUDIO CODEC ANALYSIS ===")
        print(f"Total ffmpeg audio decoders found: {len(ffmpeg_audio_codecs)}")
        print(f"Codecs: {sorted(ffmpeg_audio_codecs)}")
        
        # Jeden Codec testen
        for codec in sorted(ffmpeg_audio_codecs):
            compression_type = audio_codec_compression(codec)
            recognized_codecs[compression_type].append(codec)
            
            if compression_type == AudioCompressionBaseType.UNKNOWN:
                unknown_codecs.append(codec)
        
        # Ergebnisse ausgeben
        print("\n=== RESULTS ===")
        for compression_type, codecs in recognized_codecs.items():
            if codecs:
                print(f"\n{compression_type.value.upper()} ({len(codecs)} codecs):")
                for codec in codecs:
                    print(f"  - {codec}")
        
        if unknown_codecs:
            print(f"\n❌ UNKNOWN CODECS ({len(unknown_codecs)} codecs):")
            for codec in unknown_codecs:
                print(f"  - {codec}")
            
            # Test-Failure mit Details
            coverage_percentage = ((len(ffmpeg_audio_codecs) - len(unknown_codecs)) / len(ffmpeg_audio_codecs)) * 100
            pytest.fail(
                f"Function does not recognize {len(unknown_codecs)} out of {len(ffmpeg_audio_codecs)} "
                f"ffmpeg audio codecs ({coverage_percentage:.1f}% coverage). "
                f"Unknown codecs: {', '.join(unknown_codecs)}"
            )
        else:
            print(f"\n✅ SUCCESS: All {len(ffmpeg_audio_codecs)} ffmpeg audio codecs are recognized!")
            
        # Immer erfolgreich, wenn alle Codecs erkannt werden
        assert len(unknown_codecs) == 0, f"Unrecognized codecs: {unknown_codecs}"
    
    def test_known_codec_examples(self):
        """Testet bekannte Codec-Beispiele für Korrektheit"""
        test_cases = [
            # Uncompressed
            ("pcm_s16le", AudioCompressionBaseType.UNCOMPRESSED),
            ("pcm_f32be", AudioCompressionBaseType.UNCOMPRESSED),
            ("s16le", AudioCompressionBaseType.UNCOMPRESSED),
            
            # Lossless
            ("flac", AudioCompressionBaseType.LOSSLESS_COMPRESSED),
            ("alac", AudioCompressionBaseType.LOSSLESS_COMPRESSED),
            ("wavpack", AudioCompressionBaseType.LOSSLESS_COMPRESSED),
            
            # Lossy
            ("mp3", AudioCompressionBaseType.LOSSY_COMPRESSED),
            ("aac", AudioCompressionBaseType.LOSSY_COMPRESSED),
            ("opus", AudioCompressionBaseType.LOSSY_COMPRESSED),
            ("adpcm_ima_wav", AudioCompressionBaseType.LOSSY_COMPRESSED),
            
            # Pattern matching
            ("adpcm_ms", AudioCompressionBaseType.LOSSY_COMPRESSED),
            ("g722", AudioCompressionBaseType.LOSSY_COMPRESSED),
        ]
        
        for codec_name, expected_compression in test_cases:
            result = audio_codec_compression(codec_name)
            assert result == expected_compression, f"Codec {codec_name} should be {expected_compression}, got {result}"
    
    def test_case_insensitive_and_normalization(self):
        """Testet Normalisierung und Case-Insensitivität"""
        # Test case insensitivity
        assert audio_codec_compression("FLAC") == AudioCompressionBaseType.LOSSLESS_COMPRESSED
        assert audio_codec_compression("Mp3") == AudioCompressionBaseType.LOSSY_COMPRESSED
        assert audio_codec_compression("PCM_S16LE") == AudioCompressionBaseType.UNCOMPRESSED
        
        # Test hyphen normalization (falls relevant)
        # Dies würde funktionieren, wenn ffmpeg Codecs mit Bindestrichen hätte
        # assert audio_codec_compression("pcm-s16le") == AudioCompressionBaseType.UNCOMPRESSED


if __name__ == "__main__":
    # Kann auch direkt ausgeführt werden - zeigt Details und führt Tests aus
    print("=== Codec Analysis ===")
    codecs = get_ffmpeg_audio_decoders()
    print(f"\n-------------------------------------------\n\nFound {len(codecs)} audio decoders in ffmpeg:")
    
    # Kurze Übersicht
    compression_counts = {"uncompressed": 0, "lossless_compressed": 0, "lossy_compressed": 0, "unknown": 0}
    for codec in sorted(codecs):
        compression = audio_codec_compression(codec)
        compression_counts[compression.value] += 1
        print(f"  {codec}: {compression.value}")
    
    print(f"\n=== Summary ===")
    for comp_type, count in compression_counts.items():
        if comp_type == "unknown":
            if count == 0:
                print(f"✅ {comp_type}: {count} codecs")
            else:
                print(f"❌ {comp_type}: {count} codecs")
        else:    
            print(f"{comp_type}: {count} codecs")
    
    # Tests ausführen
    print(f"\n=== Running Tests ===")
    test_instance = TestAudioCodecCompression()
    try:
        test_instance.test_ffmpeg_codecs_coverage(codecs)
        test_instance.test_known_codec_examples()
        test_instance.test_case_insensitive_and_normalization()
        print("🎉 All tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")