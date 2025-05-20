import unittest
import pathlib
import os
import shutil
import numpy as np
import zarr
import datetime
from typing import List

# Pfade für Tests
TEST_RESULTS_DIR = pathlib.Path(__file__).parent.resolve() / "testresults"
ZARR3_STORE_DIR = TEST_RESULTS_DIR / "zarr3-store"

# Importierte Module (passe diese Pfade an deine Projektstruktur an)
from zarrwlr.aimport import (
    import_original_audio_file,
    extract_audio_segment,
    parallel_extract_audio_segments,
    check_if_original_audio_group,
    create_original_audio_group
)
from zarrwlr.config import Config

def get_test_files() -> List[pathlib.Path]:
    test_files = [
                    "testdata/audiomoth_long_snippet.wav",
                    "testdata/audiomoth_long_snippet_converted.opus",
                    "testdata/audiomoth_long_snippet_converted.flac",
                    "testdata/audiomoth_short_snippet.wav",
                    "testdata/bird1_snippet.mp3",
                    "testdata/camtrap_snippet.mov" # mp4 coded video with audio stream
                ]
    return [pathlib.Path(__file__).parent.resolve() / file for file in test_files]

def prepare_zarr_database() -> zarr.Group:
    # We don't need to prepare the root directory: LocalStore does it
    # by using root as directory name.
    store = zarr.storage.LocalStore(root=ZARR3_STORE_DIR)
    root = zarr.create_group(store=store)
    audio_import_grp = root.create_group('audio_imports')
    audio_import_grp.attrs["magic_id"] = Config.original_audio_group_magic_id
    audio_import_grp.attrs["version"] = Config.original_audio_group_version
    return audio_import_grp

class TestAudioImport(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Setup für alle Tests - Verzeichnisse erstellen und Zarr-Datenbank initialisieren"""
        if not TEST_RESULTS_DIR.exists():
            TEST_RESULTS_DIR.mkdir(parents=True)
        
        # Bestehende Zarr-Datenbank löschen, falls vorhanden
        if ZARR3_STORE_DIR.exists():
            shutil.rmtree(ZARR3_STORE_DIR)
        
        # Neue Zarr-Datenbank erstellen
        cls.zarr_group = prepare_zarr_database()
        
        # Sicherstellen, dass Testdateien existieren
        cls.test_files = get_test_files()
        missing_files = [f for f in cls.test_files if not f.exists()]
        if missing_files:
            raise FileNotFoundError(f"Folgende Testdateien fehlen: {missing_files}")
    
    def test_01_create_original_audio_group(self):
        """Test, ob die Erstellung einer Original-Audio-Gruppe funktioniert"""
        # Store-Pfad für separaten Test
        test_zarr_path = TEST_RESULTS_DIR / "test-zarr-group"
        if test_zarr_path.exists():
            shutil.rmtree(test_zarr_path)
        
        # Gruppe erstellen
        create_original_audio_group(test_zarr_path)
        
        # Prüfen, ob die Gruppe existiert und die richtigen Attribute hat
        store = zarr.DirectoryStore(test_zarr_path)
        root = zarr.open_group(store, mode='r')
        
        self.assertTrue("magic_id" in root.attrs)
        self.assertEqual(root.attrs["magic_id"], Config.original_audio_group_magic_id)
        self.assertEqual(root.attrs["version"], Config.original_audio_group_version)
        
        # Test mit Untergruppe
        group_path = "audio_test"
        create_original_audio_group(test_zarr_path, group_path)
        self.assertTrue(group_path in root)
        
        # Aufräumen
        shutil.rmtree(test_zarr_path)
    
    def test_02_import_wav_to_flac(self):
        """Test des Imports einer WAV-Datei zu FLAC"""
        # WAV-Datei nehmen
        wav_file = next(f for f in self.test_files if f.name == "audiomoth_short_snippet.wav")
        
        # Import durchführen mit aktuellem Zeitstempel
        timestamp = datetime.datetime.now()
        import_original_audio_file(
            audio_file=wav_file,
            zarr_original_audio_group=self.zarr_group,
            first_sample_time_stamp=timestamp,
            target_codec='flac',
            flac_compression_level=4
        )
        
        # Prüfen, ob eine Gruppe "0" existiert
        self.assertTrue("0" in self.zarr_group)
        
        # Prüfen, ob die Gruppe die richtigen Attribute hat
        group_0 = self.zarr_group["0"]
        self.assertEqual(group_0.attrs["type"], "original_audio_file")
        self.assertEqual(group_0.attrs["encoding"], "flac")
        
        # Prüfen, ob der Index erstellt wurde
        self.assertTrue("flac_index" in group_0)
        
        # Prüfen, ob audio_data_blob_array existiert
        self.assertTrue("audio_data_blob_array" in group_0)
        blob_array = group_0["audio_data_blob_array"]
        self.assertEqual(blob_array.attrs["codec"], "flac")
    
    def test_03_import_mp3_to_opus(self):
        """Test des Imports einer MP3-Datei zu Opus"""
        # MP3-Datei nehmen
        mp3_file = next(f for f in self.test_files if f.name == "bird1_snippet.mp3")
        
        # Import durchführen mit aktuellem Zeitstempel
        timestamp = datetime.datetime.now()
        import_original_audio_file(
            audio_file=mp3_file,
            zarr_original_audio_group=self.zarr_group,
            first_sample_time_stamp=timestamp,
            target_codec='opus',
            opus_bitrate=128000  # 128 kbps
        )
        
        # Prüfen, ob eine Gruppe "1" existiert (zweiter Import)
        self.assertTrue("1" in self.zarr_group)
        
        # Prüfen, ob die Gruppe die richtigen Attribute hat
        group_1 = self.zarr_group["1"]
        self.assertEqual(group_1.attrs["type"], "original_audio_file")
        self.assertEqual(group_1.attrs["encoding"], "opus")
        
        # Prüfen, ob der Index erstellt wurde
        self.assertTrue("ogg_page_index" in group_1)
        
        # Prüfen, ob audio_data_blob_array existiert
        self.assertTrue("audio_data_blob_array" in group_1)
        blob_array = group_1["audio_data_blob_array"]
        self.assertEqual(blob_array.attrs["codec"], "opus")
        self.assertEqual(blob_array.attrs["opus_bitrate"], 128000)
    
    def test_04_import_opus_to_opus(self):
        """Test, ob ein direkter Import von Opus zu Opus funktioniert (ohne Transkodierung)"""
        # Opus-Datei nehmen
        opus_file = next(f for f in self.test_files if f.name == "audiomoth_long_snippet_converted.opus")
        
        # Import durchführen mit aktuellem Zeitstempel
        timestamp = datetime.datetime.now()
        import_original_audio_file(
            audio_file=opus_file,
            zarr_original_audio_group=self.zarr_group,
            first_sample_time_stamp=timestamp,
            target_codec='opus'
        )
        
        # Prüfen, ob eine Gruppe "2" existiert (dritter Import)
        self.assertTrue("2" in self.zarr_group)
        
        # Prüfen, ob die Gruppe die richtigen Attribute hat
        group_2 = self.zarr_group["2"]
        self.assertEqual(group_2.attrs["encoding"], "opus")
        
        # Prüfen, ob audio_data_blob_array existiert
        self.assertTrue("audio_data_blob_array" in group_2)
        blob_array = group_2["audio_data_blob_array"]
        self.assertEqual(blob_array.attrs["codec"], "opus")
    
    def test_05_extract_audio_segment(self):
        """Test, ob das Extrahieren eines Audiosegments funktioniert"""
        # Die FLAC-Gruppe verwenden (aus test_02)
        flac_group = self.zarr_group["0"]
        
        # Segment extrahieren (ersten 1000 Samples)
        segment = extract_audio_segment(flac_group, 0, 999)
        
        # Prüfen, ob das Segment die richtige Form hat
        self.assertIsInstance(segment, np.ndarray)
        self.assertEqual(len(segment), 1000)  # 1000 Samples (inklusiv 0-999)
        
        # Die Opus-Gruppe verwenden (aus test_03)
        opus_group = self.zarr_group["1"]
        
        # Segment extrahieren (ersten 1000 Samples)
        segment = extract_audio_segment(opus_group, 0, 999)
        
        # Prüfen, ob das Segment die richtige Form hat
        self.assertIsInstance(segment, np.ndarray)
        # Opus arbeitet blockweise, daher kann die genaue Samplelänge leicht abweichen
        self.assertGreaterEqual(len(segment), 900)
    
    def test_06_parallel_extract_audio_segments(self):
        """Test, ob das parallele Extrahieren mehrerer Audiosegmente funktioniert"""
        # Die FLAC-Gruppe verwenden (aus test_02)
        flac_group = self.zarr_group["0"]
        
        # Mehrere Segmente extrahieren
        segments = [(0, 999), (1000, 1999), (2000, 2999)]
        extracted = parallel_extract_audio_segments(flac_group, segments)
        
        # Prüfen, ob die richtigen Segmente extrahiert wurden
        self.assertEqual(len(extracted), 3)
        for i, segment in enumerate(extracted):
            self.assertIsInstance(segment, np.ndarray)
            # Segmentlänge sollte 1000 sein (inklusiv)
            self.assertEqual(len(segment), 1000)
    
    def test_07_import_video_with_audio(self):
        """Test, ob das Extrahieren von Audio aus einer Videodatei funktioniert"""
        # Video-Datei mit Audiospur nehmen
        video_file = next(f for f in self.test_files if f.name == "camtrap_snippet.mov")
        
        # Import durchführen mit aktuellem Zeitstempel
        timestamp = datetime.datetime.now()
        import_original_audio_file(
            audio_file=video_file,
            zarr_original_audio_group=self.zarr_group,
            first_sample_time_stamp=timestamp,
            target_codec='flac'
        )
        
        # Prüfen, ob eine Gruppe "3" existiert (vierter Import)
        self.assertTrue("3" in self.zarr_group)
        
        # Prüfen, ob audio_data_blob_array existiert
        group_3 = self.zarr_group["3"]
        self.assertTrue("audio_data_blob_array" in group_3)
        
        # Segment extrahieren (ersten 500 Samples)
        segment = extract_audio_segment(group_3, 0, 499)
        
        # Prüfen, ob das Segment die richtige Form hat
        self.assertIsInstance(segment, np.ndarray)
        self.assertEqual(len(segment), 500)  # 500 Samples (inklusiv 0-499)
    
    def test_08_reimport_same_file(self):
        """Test, ob der Reimport derselben Datei eine Doublet-Exception auslöst"""
        # WAV-Datei nehmen (wurde bereits in test_02 importiert)
        wav_file = next(f for f in self.test_files if f.name == "audiomoth_short_snippet.wav")
        
        # Import durchführen - sollte Exception auslösen
        from zarrwlr.exceptions import Doublet
        with self.assertRaises(Doublet):
            import_original_audio_file(
                audio_file=wav_file,
                zarr_original_audio_group=self.zarr_group,
                first_sample_time_stamp=datetime.datetime.now(),
                target_codec='flac'
            )
    
    @classmethod
    def tearDownClass(cls):
        """Aufräumen nach allen Tests"""
        # Hier können wir entscheiden, ob die Testdatenbank erhalten bleiben soll oder nicht
        # Wenn du sie behalten willst zum manuellen Inspizieren, kommentiere die nächste Zeile aus:
        # shutil.rmtree(ZARR3_STORE_DIR)
        pass


if __name__ == "__main__":
    unittest.main()
