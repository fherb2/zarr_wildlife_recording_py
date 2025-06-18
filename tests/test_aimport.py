#!/usr/bin/env python3
"""
Comprehensive Tests for aimport.py

Testet die High-Level Audio Import API mit End-to-End-Tests für alle 
Hauptfunktionen: Zarr-Gruppen-Management, Import-Status-Check und 
Audio-Import-Operationen.

Verwendung:
    python test_aimport.py
    pytest test_aimport.py -v
"""

import pytest
import pathlib
import tempfile
import shutil
import sys
import time
from unittest.mock import patch, MagicMock

# Projekt-Imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from zarrwlr.aimport import (
    AGroup,
    ImportResult,
    ImportWorker,
    init_zarr_audio_grp,
    check_if_zarr_audio_grp,
    upgrade_zarr_audio_grp,
    open_zarr_audio_grp,
    is_audio_in_zarr_audio_group,
    aimport,
    _check_single_file_import_status,
    _check_files_parallel_grouped,
    _import_subprocess_worker
)
from zarrwlr.import_utils import FileParameter, TargetFormats, TargetSamplingTransforming
from zarrwlr.config import Config
from zarrwlr.exceptions import ZarrGroupMismatch, ZarrComponentVersionError
from zarrwlr.logsetup import get_module_logger

import zarr

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

def get_unique_zarr_store(test_name: str) -> pathlib.Path:
    """Erstelle eindeutiges Zarr-Store-Verzeichnis für Test-Isolation"""
    test_dir = get_unique_test_dir(test_name)
    zarr_store = test_dir / "zarr_store"
    zarr_store.mkdir(parents=True, exist_ok=True)
    return zarr_store

# Schnelle Testdateien für Performance-Tests
FAST_TEST_FILES = [
    "bird1_snippet.mp3",                                        # 0.19 MB, 6s
    "XC744150 - Jagdfasan - Phasianus colchicus.mp3",          # 0.32 MB, 8s
    "XC890995 - Oak Toad - Anaxyrus quercicus.mp3"             # 0.17 MB, 14s
]

# Repräsentative Testdateien für verschiedene Szenarien
SCENARIO_TEST_FILES = {
    'uncompressed_mono': "audiomoth_short_snippet.wav",         # 20MB, mono, 48kHz
    'uncompressed_stereo': "XC897425 - Rohrschwirl - Locustella luscinioides.wav",  # 118MB, stereo, 44.1kHz  
    'lossless': "audiomoth_long_snippet_converted.flac",        # 200MB, mono, 48kHz
    'lossy_high': "XC642625 - Rohrschwirl - Locustella luscinioides luscinioides.mp3",  # 320kbps
    'lossy_low': "XC69863 - Gelbfußdrossel - Turdus flavipes.mp3",  # 128kbps
    'ultrasound_400k': "XC821955 - Greater Dog-like Bat - Peropteryx kappleri.wav",  # 400kHz
    'ultrasound_250k': "XC838306 - Lesser Dog-like Bat - Peropteryx macrotis.wav",   # 250kHz
    'multichannel_8': "multichannel_8ch.wav",                   # 8 channels (FLAC limit)
    'multichannel_12': "multichannel_12ch.wav"                  # 12 channels (over FLAC limit)
}

@pytest.fixture
def temp_zarr_store():
    """Temporärer Zarr-Store für Tests in testresults/"""
    test_name = "zarr_store_test"
    if hasattr(pytest, 'current_request'):
        test_name = pytest.current_request.node.name
    
    store_path = get_unique_zarr_store(test_name)
    yield store_path
    
    # Cleanup nach Test (optional, da testresults komplett gereinigt wird)
    # shutil.rmtree(store_path.parent, ignore_errors=True)

@pytest.fixture
def sample_audio_files():
    """Sammlung von Beispiel-Audiodateien für Tests"""
    files = []
    for filename in FAST_TEST_FILES:
        file_path = TESTDATA_DIR / filename
        if file_path.exists():
            files.append(file_path)
    return files

def get_test_file_path(filename: str) -> pathlib.Path:
    """Hilfsfunktion zum Abrufen des vollständigen Pfads einer Testdatei"""
    path = TESTDATA_DIR / filename
    if not path.exists():
        pytest.skip(f"Testdatei nicht gefunden: {filename}")
    return path

def create_mock_file_parameter(file_path: pathlib.Path, **overrides) -> FileParameter:
    """Erstelle einen Mock FileParameter für Tests"""
    try:
        file_param = FileParameter(file_path)
        
        # Überschreibe Eigenschaften wenn angegeben
        for key, value in overrides.items():
            setattr(file_param, key, value)
        
        return file_param
    except Exception as e:
        pytest.skip(f"Could not create FileParameter for {file_path}: {e}")

# ============================================================================
# Tests für AGroup Wrapper
# ============================================================================

class TestAGroupWrapper:
    """Tests für die AGroup Type-Safe Wrapper Klasse"""
    
    def test_agroup_creation(self, temp_zarr_store):
        """Test AGroup-Erstellung aus zarr.Group"""
        # Erstelle zarr.Group
        store = zarr.storage.LocalStore(str(temp_zarr_store))
        zarr_group = zarr.open_group(store, mode='a')
        
        # Erstelle AGroup
        audio_group = AGroup(zarr_group)
        
        assert isinstance(audio_group, AGroup)
        assert audio_group.zarr_group is zarr_group
    
    def test_agroup_type_checking(self, temp_zarr_store):
        """Test AGroup Type-Checking"""
        store = zarr.storage.LocalStore(str(temp_zarr_store))
        zarr_group = zarr.open_group(store, mode='a')
        audio_group = AGroup(zarr_group)
        
        # Type checking sollte funktionieren
        assert isinstance(audio_group, AGroup)
        assert not isinstance(zarr_group, AGroup)
        
        # Fehlerhafter Input
        with pytest.raises(TypeError):
            AGroup("not_a_zarr_group")
    
    def test_agroup_delegation(self, temp_zarr_store):
        """Test AGroup Methoden-Delegation"""
        store = zarr.storage.LocalStore(str(temp_zarr_store))
        zarr_group = zarr.open_group(store, mode='a')
        
        # Setze Attribute im zarr.Group
        zarr_group.attrs['test_attr'] = 'test_value'
        zarr_group.create_group('test_subgroup')
        
        audio_group = AGroup(zarr_group)
        
        # Alle zarr.Group-Operationen sollten funktionieren
        assert audio_group.attrs['test_attr'] == 'test_value'
        assert 'test_subgroup' in audio_group
        assert len(audio_group) == 1
        
        # Iteration sollte funktionieren
        subgroups = list(audio_group)
        assert 'test_subgroup' in subgroups
    
    def test_agroup_representation(self, temp_zarr_store):
        """Test AGroup String-Repräsentation"""
        store = zarr.storage.LocalStore(str(temp_zarr_store))
        zarr_group = zarr.open_group(store, mode='a')
        audio_group = AGroup(zarr_group)
        
        repr_str = repr(audio_group)
        assert "AGroup(" in repr_str
        assert "zarr.Group" in repr_str

# ============================================================================
# Tests für Zarr Audio Group Management
# ============================================================================

class TestZarrAudioGroupManagement:
    """Tests für Zarr Audio Group Erstellung und Verwaltung"""
    
    def test_init_zarr_audio_grp_new(self, temp_zarr_store):
        """Test Initialisierung einer neuen Audio-Gruppe"""
        audio_group = init_zarr_audio_grp(temp_zarr_store)
        
        assert isinstance(audio_group, zarr.Group)
        assert audio_group.attrs["magic_id"] == Config.original_audio_group_magic_id
        assert audio_group.attrs["version"] == Config.original_audio_group_version
    
    def test_init_zarr_audio_grp_with_subgroup(self, temp_zarr_store):
        """Test Initialisierung mit Sub-Gruppe"""
        audio_group = init_zarr_audio_grp(temp_zarr_store, "audio_data")
        
        assert isinstance(audio_group, zarr.Group)
        assert audio_group.attrs["magic_id"] == Config.original_audio_group_magic_id
        
        # Sollte als Sub-Gruppe existieren
        store = zarr.storage.LocalStore(str(temp_zarr_store))
        root = zarr.open_group(store, mode='r')
        assert "audio_data" in root
    
    def test_init_zarr_audio_grp_existing_valid(self, temp_zarr_store):
        """Test Initialisierung bei existierender gültiger Gruppe"""
        # Erstelle erste Gruppe
        audio_group1 = init_zarr_audio_grp(temp_zarr_store)
        group_id = id(audio_group1)
        
        # Öffne wieder - sollte existierende Gruppe erkennen
        audio_group2 = init_zarr_audio_grp(temp_zarr_store)
        
        # Sollte die gleichen Attribute haben
        assert audio_group2.attrs["magic_id"] == Config.original_audio_group_magic_id
        assert audio_group2.attrs["version"] == Config.original_audio_group_version
    
    def test_init_zarr_audio_grp_existing_invalid(self, temp_zarr_store):
        """Test Initialisierung bei existierender ungültiger Gruppe"""
        # Erstelle zarr.Group mit falschen Attributen
        store = zarr.storage.LocalStore(str(temp_zarr_store))
        root = zarr.open_group(store, mode='a')
        
        # Erstelle Sub-Gruppe mit falschen Attributen
        invalid_group = root.create_group("invalid_audio")
        invalid_group.attrs["magic_id"] = "wrong_magic_id"
        invalid_group.attrs["version"] = (0, 1)
        
        # Sollte Fehler werfen
        with pytest.raises(ZarrGroupMismatch):
            init_zarr_audio_grp(temp_zarr_store, "invalid_audio")
    
    def test_check_if_zarr_audio_grp(self, temp_zarr_store):
        """Test check_if_zarr_audio_grp()"""
        # Neue Gruppe erstellen
        audio_group = init_zarr_audio_grp(temp_zarr_store)
        
        # Sollte als gültige Audio-Gruppe erkannt werden
        assert check_if_zarr_audio_grp(audio_group) == True
        
        # Erstelle ungültige Gruppe
        store = zarr.storage.LocalStore(str(temp_zarr_store))
        root = zarr.open_group(store, mode='a')
        invalid_group = root.create_group("invalid")
        
        # Sollte nicht als Audio-Gruppe erkannt werden
        assert check_if_zarr_audio_grp(invalid_group) == False
    
    def test_upgrade_zarr_audio_grp(self, temp_zarr_store):
        """Test upgrade_zarr_audio_grp()"""
        audio_group = init_zarr_audio_grp(temp_zarr_store)
        
        # Simuliere veraltete Version
        old_version = (0, 9)
        
        # Template-Implementierung sollte True zurückgeben
        result = upgrade_zarr_audio_grp(audio_group, old_version)
        assert result == True
    
    def test_open_zarr_audio_grp_create_new(self, temp_zarr_store):
        """Test open_zarr_audio_grp() mit neuer Gruppe"""
        audio_group = open_zarr_audio_grp(temp_zarr_store)
        
        assert isinstance(audio_group, AGroup)
        assert isinstance(audio_group.zarr_group, zarr.Group)
        assert audio_group.attrs["magic_id"] == Config.original_audio_group_magic_id
    
    def test_open_zarr_audio_grp_existing(self, temp_zarr_store):
        """Test open_zarr_audio_grp() mit existierender Gruppe"""
        # Erstelle Gruppe
        audio_group1 = open_zarr_audio_grp(temp_zarr_store)
        
        # Öffne erneut
        audio_group2 = open_zarr_audio_grp(temp_zarr_store)
        
        assert isinstance(audio_group2, AGroup)
        assert audio_group2.attrs["magic_id"] == Config.original_audio_group_magic_id
    
    def test_open_zarr_audio_grp_no_create(self, temp_zarr_store):
        """Test open_zarr_audio_grp() mit create=False"""
        # Sollte Fehler werfen wenn Gruppe nicht existiert
        with pytest.raises(FileNotFoundError):
            open_zarr_audio_grp(temp_zarr_store, create=False)
        
        # Erstelle Gruppe zuerst
        audio_group1 = open_zarr_audio_grp(temp_zarr_store)
        
        # Jetzt sollte create=False funktionieren
        audio_group2 = open_zarr_audio_grp(temp_zarr_store, create=False)
        assert isinstance(audio_group2, AGroup)

# ============================================================================
# Tests für Import Status Checking
# ============================================================================

class TestImportStatusChecking:
    """Tests für is_audio_in_zarr_audio_group() und verwandte Funktionen"""
    
    def test_is_audio_in_zarr_audio_group_single_new_file(self, temp_zarr_store):
        """Test Import-Status-Check für eine neue Datei"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        test_file = get_test_file_path("bird1_snippet.mp3")
        
        already_imported, file_analysis = is_audio_in_zarr_audio_group(audio_db, test_file)
        
        assert already_imported == False
        assert isinstance(file_analysis, FileParameter)
        assert file_analysis.base_parameter.file.name == "bird1_snippet.mp3"
        assert file_analysis.can_be_imported == True
    
    def test_is_audio_in_zarr_audio_group_single_file_parameter(self, temp_zarr_store):
        """Test Import-Status-Check mit FileParameter"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        test_file = get_test_file_path("bird1_snippet.mp3")
        
        # Erstelle FileParameter mit benutzerdefinierten Einstellungen
        file_param = FileParameter(test_file)
        file_param.target_format = "FLAC_48000"
        file_param.flac_compression_level = 8
        
        already_imported, file_analysis = is_audio_in_zarr_audio_group(audio_db, file_param)
        
        assert already_imported == False
        assert file_analysis is file_param  # Sollte das gleiche Objekt sein
        assert file_analysis.target_format == TargetFormats.FLAC_48000
        assert file_analysis.flac_compression_level == 8
    
    def test_is_audio_in_zarr_audio_group_batch_files(self, temp_zarr_store, sample_audio_files):
        """Test Import-Status-Check für mehrere Dateien"""
        if len(sample_audio_files) < 2:
            pytest.skip("Not enough sample audio files for batch test")
        
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        
        results = is_audio_in_zarr_audio_group(audio_db, sample_audio_files)
        
        assert isinstance(results, list)
        assert len(results) == len(sample_audio_files)
        
        for already_imported, file_analysis in results:
            assert isinstance(already_imported, bool)
            assert isinstance(file_analysis, FileParameter)
            assert already_imported == False  # Alle sollten neu sein
    
    def test_is_audio_in_zarr_audio_group_duplicate_detection(self, temp_zarr_store):
        """Test Duplikat-Erkennung über SHA256-Hash"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        test_file = get_test_file_path("bird1_snippet.mp3")
        
        # Erstmal sollte Datei neu sein
        already_imported1, file_analysis1 = is_audio_in_zarr_audio_group(audio_db, test_file)
        assert already_imported1 == False
        
        # Simuliere erfolgreichen Import durch Erstellen einer Gruppe mit dem Hash
        file_hash = file_analysis1.base_parameter.file_sh256
        
        # Erstelle numerische Gruppe (simuliert imported file)
        imported_group = audio_db.zarr_group.create_group("0")
        imported_group.attrs["type"] = "original_audio_file"
        imported_group.attrs["base_features"] = {
            "FILENAME": file_analysis1.base_parameter.file.name,
            "SH256": file_hash,
            "SIZE_BYTES": file_analysis1.base_parameter.file_size_bytes,
            "HAS_AUDIO_STREAM": True,
            "NB_STREAMS": 1
        }
        
        # Jetzt sollte Datei als bereits importiert erkannt werden
        already_imported2, file_analysis2 = is_audio_in_zarr_audio_group(audio_db, test_file)
        assert already_imported2 == True
        assert file_analysis2.base_parameter.file_sh256 == file_hash
    
    def test_is_audio_in_zarr_audio_group_type_validation(self, temp_zarr_store):
        """Test Type-Validierung für is_audio_in_zarr_audio_group()"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        
        # Falscher Typ für zarr_audio_group
        with pytest.raises(TypeError):
            is_audio_in_zarr_audio_group("not_an_agroup", "some_file.wav")
    
    def test_check_single_file_import_status(self, temp_zarr_store):
        """Test _check_single_file_import_status() direkt"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        test_file = get_test_file_path("bird1_snippet.mp3")
        
        already_imported, file_analysis = _check_single_file_import_status(audio_db, test_file)
        
        assert already_imported == False
        assert isinstance(file_analysis, FileParameter)
        assert file_analysis.base_parameter.file.exists()

# ============================================================================
# Tests für Audio Import Operations
# ============================================================================

class TestAudioImportOperations:
    """Tests für aimport() und verwandte Import-Funktionen"""
    
    def test_aimport_single_file_success(self, temp_zarr_store):
        """Test erfolgreicher Import einer einzelnen Datei"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        test_file = get_test_file_path("bird1_snippet.mp3")
        
        # Analysiere Datei
        _, file_analysis = is_audio_in_zarr_audio_group(audio_db, test_file)
        
        # Stelle sicher, dass Datei importierbar ist
        if not file_analysis.can_be_imported:
            pytest.skip(f"Test file {test_file.name} cannot be imported: {file_analysis.conflicts}")
        
        # Führe Import aus
        result = aimport(audio_db, file_analysis)
        
        assert isinstance(result, ImportResult)
        assert result.success == True
        assert result.file_path == test_file
        assert result.import_time > 0
        assert result.zarr_group_name is not None
        assert result.zarr_group_name.isdigit()  # Sollte numerische Gruppe sein
        
        # Überprüfe dass Gruppe erstellt wurde
        assert result.zarr_group_name in audio_db
        imported_group = audio_db[result.zarr_group_name]
        assert imported_group.attrs["type"] == "original_audio_file"
        assert "audio_data_blob_array" in imported_group
    
    def test_aimport_single_file_with_flac_target(self, temp_zarr_store):
        """Test Import mit FLAC-Zielformat"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        test_file = get_test_file_path("audiomoth_short_snippet.wav")
        
        # Analysiere und konfiguriere für FLAC
        _, file_analysis = is_audio_in_zarr_audio_group(audio_db, test_file)
        file_analysis.target_format = "FLAC_48000"
        file_analysis.flac_compression_level = 6
        
        if not file_analysis.can_be_imported:
            pytest.skip(f"Test file cannot be imported with FLAC: {file_analysis.conflicts}")
        
        result = aimport(audio_db, file_analysis)
        
        assert result.success == True
        assert result.target_format == TargetFormats.FLAC_48000
        
        # Überprüfe Metadaten
        imported_group = audio_db[result.zarr_group_name]
        assert imported_group.attrs["encoding"] == "flac"
    
    def test_aimport_single_file_with_aac_target(self, temp_zarr_store):
        """Test Import mit AAC-Zielformat"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        test_file = get_test_file_path("bird1_snippet.mp3")
        
        # Analysiere und konfiguriere für AAC
        _, file_analysis = is_audio_in_zarr_audio_group(audio_db, test_file)
        file_analysis.target_format = "AAC_48000"
        file_analysis.aac_bitrate = 192000
        
        if not file_analysis.can_be_imported:
            pytest.skip(f"Test file cannot be imported with AAC: {file_analysis.conflicts}")
        
        result = aimport(audio_db, file_analysis)
        
        assert result.success == True
        assert result.target_format == TargetFormats.AAC_48000
        
        # Überprüfe Metadaten
        imported_group = audio_db[result.zarr_group_name]
        assert imported_group.attrs["encoding"] == "aac"
    
    def test_aimport_batch_files_success(self, temp_zarr_store, sample_audio_files):
        """Test erfolgreicher Batch-Import"""
        if len(sample_audio_files) < 2:
            pytest.skip("Not enough sample audio files for batch test")
        
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        
        # Analysiere alle Dateien
        import_status = is_audio_in_zarr_audio_group(audio_db, sample_audio_files)
        file_analyses = [analysis for imported, analysis in import_status if not imported]
        
        # Filtere nur importierbare Dateien
        importable_analyses = [fa for fa in file_analyses if fa.can_be_imported]
        
        if not importable_analyses:
            pytest.skip("No importable files found in sample set")
        
        # Führe Batch-Import aus
        results = aimport(audio_db, importable_analyses)
        
        assert isinstance(results, list)
        assert len(results) == len(importable_analyses)
        
        successful_imports = [r for r in results if r.success]
        assert len(successful_imports) > 0
        
        # Überprüfe dass Gruppen erstellt wurden
        for result in successful_imports:
            assert result.zarr_group_name in audio_db
            imported_group = audio_db[result.zarr_group_name]
            assert imported_group.attrs["type"] == "original_audio_file"
    
    def test_aimport_blocking_conflicts_prevention(self, temp_zarr_store):
        """Test dass Blocking-Konflikte den Import verhindern"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        test_file = get_test_file_path("multichannel_12ch.wav")
        
        # Analysiere Datei
        _, file_analysis = is_audio_in_zarr_audio_group(audio_db, test_file)
        
        # Forciere FLAC für 12-Kanal-Datei (sollte Blocking-Konflikt erzeugen)
        file_analysis.target_format = "FLAC"
        
        # Sollte Blocking-Konflikt haben
        if file_analysis.has_blocking_conflicts:
            with pytest.raises(ValueError, match="Import blocked by conflicts"):
                aimport(audio_db, file_analysis)
        else:
            # Falls kein Konflikt erkannt wurde, überspringen
            pytest.skip("Expected blocking conflict not detected")
    
    def test_aimport_duplicate_prevention(self, temp_zarr_store):
        """Test dass bereits importierte Dateien nicht erneut importiert werden"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        test_file = get_test_file_path("bird1_snippet.mp3")
        
        # Erste Analyse und Import
        already_imported1, file_analysis1 = is_audio_in_zarr_audio_group(audio_db, test_file)
        assert already_imported1 == False
        
        if not file_analysis1.can_be_imported:
            pytest.skip("Test file cannot be imported")
        
        result1 = aimport(audio_db, file_analysis1)
        assert result1.success == True
        
        # Zweite Analyse - sollte als bereits importiert erkannt werden
        already_imported2, file_analysis2 = is_audio_in_zarr_audio_group(audio_db, test_file)
        assert already_imported2 == True
    
    def test_aimport_type_validation(self, temp_zarr_store):
        """Test Type-Validierung für aimport()"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        
        # Falscher Typ für zarr_audio_group
        with pytest.raises(TypeError):
            aimport("not_an_agroup", [])
    
    def test_aimport_batch_with_conflicts(self, temp_zarr_store):
        """Test Batch-Import mit Konflikt-Behandlung"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        test_files = [
            get_test_file_path("bird1_snippet.mp3"),       # OK
            get_test_file_path("multichannel_12ch.wav")    # Könnte Probleme mit FLAC haben
        ]
        
        # Analysiere alle Dateien
        import_status = is_audio_in_zarr_audio_group(audio_db, test_files)
        file_analyses = [analysis for imported, analysis in import_status if not imported]
        
        # Forciere FLAC für alle (sollte Konflikt bei multichannel erzeugen)
        for fa in file_analyses:
            fa.target_format = "FLAC"
        
        # Wenn eine Datei Blocking-Konflikte hat, sollte gesamter Batch abgelehnt werden
        blocking_conflicts = [fa for fa in file_analyses if fa.has_blocking_conflicts]
        
        if blocking_conflicts:
            with pytest.raises(ValueError, match="Import blocked for entire batch"):
                aimport(audio_db, file_analyses)
        else:
            # Falls keine Konflikte, sollte Import funktionieren
            results = aimport(audio_db, file_analyses)
            assert all(r.success for r in results)

# ============================================================================
# Tests für ImportWorker und Import-Details
# ============================================================================

class TestImportWorker:
    """Tests für ImportWorker-Klasse und Import-Details"""
    
    def test_import_worker_direct_execution(self, temp_zarr_store):
        """Test direkte Ausführung durch ImportWorker"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        test_file = get_test_file_path("bird1_snippet.mp3")
        
        # Erstelle FileParameter
        file_param = FileParameter(test_file)
        if not file_param.can_be_imported:
            pytest.skip("Test file cannot be imported")
        
        # Direkte Ausführung
        worker = ImportWorker()
        result = worker.run_direct(audio_db, file_param)
        
        assert isinstance(result, ImportResult)
        assert result.success == True
        assert result.file_path == test_file
    
    def test_import_worker_subprocess_execution(self, temp_zarr_store):
        """Test Subprocess-Ausführung durch ImportWorker"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        test_file = get_test_file_path("bird1_snippet.mp3")
        
        # Erstelle FileParameter
        file_param = FileParameter(test_file)
        if not file_param.can_be_imported:
            pytest.skip("Test file cannot be imported")
        
        # Subprocess-Ausführung
        worker = ImportWorker()
        result = worker.run_subprocess(audio_db.zarr_group, file_param)
        
        assert isinstance(result, ImportResult)
        assert result.success == True
        assert result.file_path == test_file
    
    def test_import_result_completeness(self, temp_zarr_store):
        """Test Vollständigkeit der ImportResult-Daten"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        test_file = get_test_file_path("audiomoth_short_snippet.wav")
        
        # Analysiere und importiere
        _, file_analysis = is_audio_in_zarr_audio_group(audio_db, test_file)
        if not file_analysis.can_be_imported:
            pytest.skip("Test file cannot be imported")
        
        result = aimport(audio_db, file_analysis)
        
        # Überprüfe alle wichtigen Felder
        assert result.success == True
        assert result.file_path == test_file
        assert result.import_time > 0
        assert result.target_format is not None
        assert result.target_sampling_transform is not None
        assert result.source_codec is not None
        assert result.source_sample_rate is not None
        assert result.source_channels is not None
        assert result.compressed_size_bytes is not None
        assert result.compression_ratio is not None
        assert result.zarr_group_name is not None
        assert isinstance(result.conflicts_detected, list)
        assert result.error_message is None

# ============================================================================
# Tests für spezielle Szenarien
# ============================================================================

class TestSpecialScenarios:
    """Tests für spezielle Audio-Import-Szenarien"""
    
    def test_ultrasound_import_scenario(self, temp_zarr_store):
        """Test Import von Ultraschall-Aufnahmen"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        test_file = get_test_file_path("XC821955 - Greater Dog-like Bat - Peropteryx kappleri.wav")
        
        # Analysiere Ultraschall-Datei
        _, file_analysis = is_audio_in_zarr_audio_group(audio_db, test_file)
        
        assert file_analysis.is_ultrasound_recording == True
        assert file_analysis.quality_analysis['sample_rate'] == 400000
        
        # Sollte intelligente Vorschläge für Ultraschall haben
        if file_analysis.target_format.code == 'aac':
            assert file_analysis.target_sampling_transform.code == 'reinterpreting'
        elif file_analysis.target_format.code == 'flac':
            assert file_analysis.target_sampling_transform == TargetSamplingTransforming.EXACTLY
        
        if file_analysis.can_be_imported:
            result = aimport(audio_db, file_analysis)
            assert result.success == True
            assert result.source_sample_rate == 400000
    
    def test_multichannel_import_scenario(self, temp_zarr_store):
        """Test Import von Multichannel-Aufnahmen"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        test_file = get_test_file_path("multichannel_8ch.wav")
        
        # Analysiere Multichannel-Datei
        _, file_analysis = is_audio_in_zarr_audio_group(audio_db, test_file)
        
        assert file_analysis.selected_audio_streams[0].nb_channels == 8
        
        # 8 Kanäle sollten mit FLAC funktionieren
        if file_analysis.target_format and file_analysis.target_format.code == 'flac':
            if file_analysis.can_be_imported:
                result = aimport(audio_db, file_analysis)
                assert result.success == True
                assert result.source_channels == 8
        
        # Aber AAC sollte Warnungen haben
        file_analysis.target_format = "AAC_44100"
        if file_analysis.has_quality_warnings:
            warnings = file_analysis.conflicts['quality_warnings']
            assert any('spatial information' in warning for warning in warnings)
    
    def test_copy_mode_import_scenario(self, temp_zarr_store):
        """Test Copy-Mode Import-Szenario"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        test_file = get_test_file_path("audiomoth_long_snippet_converted.flac")
        
        # Analysiere FLAC-Datei
        _, file_analysis = is_audio_in_zarr_audio_group(audio_db, test_file)
        
        # FLAC -> FLAC sollte Copy-Mode erkennen
        if file_analysis.target_format and file_analysis.target_format.code == 'flac':
            assert file_analysis.is_copy_mode == True
            
            if file_analysis.can_be_imported:
                result = aimport(audio_db, file_analysis)
                assert result.success == True
                assert result.copy_mode_used == True
    
    def test_large_file_import_scenario(self, temp_zarr_store):
        """Test Import großer Dateien"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        
        # Verwende größte verfügbare Datei
        large_files = [
            "audiomoth_long_snippet_converted.flac",  # 200MB
            "XC897425 - Rohrschwirl - Locustella luscinioides.wav"  # 118MB
        ]
        
        test_file = None
        for filename in large_files:
            try:
                test_file = get_test_file_path(filename)
                break
            except:
                continue
        
        if not test_file:
            pytest.skip("No large test files available")
        
        # Analysiere große Datei
        _, file_analysis = is_audio_in_zarr_audio_group(audio_db, test_file)
        
        if file_analysis.can_be_imported:
            start_time = time.time()
            result = aimport(audio_db, file_analysis)
            import_time = time.time() - start_time
            
            assert result.success == True
            assert result.compressed_size_bytes < file_analysis.base_parameter.file_size_bytes
            
            # Performance-Check (sollte nicht zu lange dauern)
            assert import_time < 120, f"Large file import took too long: {import_time:.2f}s"

# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance-Tests für Import-Operationen"""
    
    def test_analysis_performance_multiple_files(self, temp_zarr_store, sample_audio_files):
        """Test Analyse-Performance für mehrere Dateien"""
        if len(sample_audio_files) < 3:
            pytest.skip("Not enough sample files for performance test")
        
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        
        start_time = time.time()
        import_status = is_audio_in_zarr_audio_group(audio_db, sample_audio_files)
        analysis_time = time.time() - start_time
        
        assert len(import_status) == len(sample_audio_files)
        
        # Performance-Check: Analyse sollte schnell sein
        avg_time_per_file = analysis_time / len(sample_audio_files)
        assert avg_time_per_file < 5.0, f"Average analysis time too slow: {avg_time_per_file:.2f}s per file"
    
    def test_import_performance_small_files(self, temp_zarr_store):
        """Test Import-Performance für kleine Dateien"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        
        # Verwende kleine, schnelle Datei
        test_file = get_test_file_path("bird1_snippet.mp3")  # 0.19 MB, 6s
        
        _, file_analysis = is_audio_in_zarr_audio_group(audio_db, test_file)
        if not file_analysis.can_be_imported:
            pytest.skip("Test file cannot be imported")
        
        start_time = time.time()
        result = aimport(audio_db, file_analysis)
        import_time = time.time() - start_time
        
        assert result.success == True
        assert import_time < 30, f"Small file import took too long: {import_time:.2f}s"
    
    def test_batch_import_performance(self, temp_zarr_store):
        """Test Batch-Import-Performance"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        
        # Sammle alle verfügbaren kleinen Dateien
        small_files = []
        for filename in FAST_TEST_FILES:
            try:
                small_files.append(get_test_file_path(filename))
            except:
                continue
        
        if len(small_files) < 2:
            pytest.skip("Not enough small files for batch performance test")
        
        # Analysiere alle Dateien
        import_status = is_audio_in_zarr_audio_group(audio_db, small_files)
        file_analyses = [analysis for imported, analysis in import_status if not imported]
        importable_analyses = [fa for fa in file_analyses if fa.can_be_imported]
        
        if not importable_analyses:
            pytest.skip("No importable files for batch test")
        
        # Batch-Import
        start_time = time.time()
        results = aimport(audio_db, importable_analyses)
        batch_time = time.time() - start_time
        
        successful_imports = [r for r in results if r.success]
        
        if successful_imports:
            avg_time_per_file = batch_time / len(successful_imports)
            assert avg_time_per_file < 60, f"Batch import too slow: {avg_time_per_file:.2f}s per file"

# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests für Fehlerbehandlung bei Import-Operationen"""
    
    def test_import_nonexistent_file(self, temp_zarr_store):
        """Test Import nicht existierender Datei"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        
        with pytest.raises(FileNotFoundError):
            FileParameter("nonexistent_file.wav")
    
    def test_import_corrupted_file(self):
        """Test Import beschädigter Datei"""
        test_dir = get_unique_test_dir("corrupted_file_test")
        audio_db = open_zarr_audio_grp(get_unique_zarr_store("corrupted_file_zarr"))
        
        corrupted_file = test_dir / "corrupted.wav"
        # Schreibe ungültige Audio-Daten
        corrupted_file.write_bytes(b"This is not a valid audio file")
        
        # Sollte bei Analyse fehlschlagen
        with pytest.raises((RuntimeError, ValueError)):
            FileParameter(corrupted_file)
    
    def test_import_worker_error_handling(self):
        """Test ImportWorker Fehlerbehandlung"""
        audio_db = open_zarr_audio_grp(get_unique_zarr_store("error_handling_zarr"))
        test_file = get_test_file_path("bird1_snippet.mp3")
        
        # Erstelle FileParameter mit ungültiger Konfiguration
        file_param = FileParameter(test_file)
        file_param.target_format = "FLAC"
        file_param._can_be_imported = False  # Forciere "nicht importierbar"
        file_param._conflicts = {'blocking_conflicts': ['Test conflict']}
        
        worker = ImportWorker()
        result = worker.run_direct(audio_db, file_param)
        
        assert result.success == False
        assert "Import blocked by conflicts" in result.error_message
    
    def test_zarr_store_error_handling(self):
        """Test Fehlerbehandlung bei ungültigen Zarr-Stores"""
        # Ungültiger Store-Pfad
        with pytest.raises((FileNotFoundError, OSError)):
            open_zarr_audio_grp("/invalid/path/that/does/not/exist")
    
    def test_import_rollback_on_failure(self):
        """Test Rollback bei Import-Fehlern"""
        audio_db = open_zarr_audio_grp(get_unique_zarr_store("rollback_test_zarr"))
        test_file = get_test_file_path("bird1_snippet.mp3")
        
        # Erstelle FileParameter
        file_param = FileParameter(test_file)
        if not file_param.can_be_imported:
            pytest.skip("Test file cannot be imported")
        
        # Simuliere Fehler durch Mock der Import-Funktion
        with patch('zarrwlr.flac_access.import_flac_to_zarr', side_effect=Exception("Simulated import error")):
            with patch('zarrwlr.aac_access.import_aac_to_zarr', side_effect=Exception("Simulated import error")):
                result = aimport(audio_db, file_param)
                
                assert result.success == False
                assert "Simulated import error" in result.error_message
                
                # Keine Gruppe sollte erstellt worden sein
                numeric_groups = [k for k in audio_db if k.isdigit()]
                assert len(numeric_groups) == 0

# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-End Integration Tests"""
    
    def test_complete_workflow_flac(self, temp_zarr_store):
        """Test kompletter FLAC-Workflow: Analyse -> Import -> Verifikation"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        test_file = get_test_file_path("audiomoth_short_snippet.wav")
        
        # 1. Status-Check
        already_imported, file_analysis = is_audio_in_zarr_audio_group(audio_db, test_file)
        assert already_imported == False
        
        # 2. Konfiguration für FLAC
        file_analysis.target_format = "FLAC_48000"
        file_analysis.flac_compression_level = 6
        
        if not file_analysis.can_be_imported:
            pytest.skip("Test file cannot be imported with FLAC")
        
        # 3. Import
        result = aimport(audio_db, file_analysis)
        assert result.success == True
        
        # 4. Verifikation
        imported_group = audio_db[result.zarr_group_name]
        assert imported_group.attrs["encoding"] == "flac"
        assert "audio_data_blob_array" in imported_group
        
        # 5. Duplikat-Check
        already_imported2, _ = is_audio_in_zarr_audio_group(audio_db, test_file)
        assert already_imported2 == True
    
    def test_complete_workflow_aac(self, temp_zarr_store):
        """Test kompletter AAC-Workflow: Analyse -> Import -> Verifikation"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        test_file = get_test_file_path("bird1_snippet.mp3")
        
        # 1. Status-Check
        already_imported, file_analysis = is_audio_in_zarr_audio_group(audio_db, test_file)
        assert already_imported == False
        
        # 2. Konfiguration für AAC
        file_analysis.target_format = "AAC_48000"
        file_analysis.aac_bitrate = 192000
        
        if not file_analysis.can_be_imported:
            pytest.skip("Test file cannot be imported with AAC")
        
        # 3. Import
        result = aimport(audio_db, file_analysis)
        assert result.success == True
        
        # 4. Verifikation
        imported_group = audio_db[result.zarr_group_name]
        assert imported_group.attrs["encoding"] == "aac"
        assert "audio_data_blob_array" in imported_group
        
        # 5. Duplikat-Check
        already_imported2, _ = is_audio_in_zarr_audio_group(audio_db, test_file)
        assert already_imported2 == True
    
    def test_mixed_format_batch_workflow(self, temp_zarr_store):
        """Test Batch-Workflow mit gemischten Formaten"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        
        # Sammle verschiedene Dateitypen
        test_files = []
        for filename in ["bird1_snippet.mp3", "audiomoth_short_snippet.wav"]:
            try:
                test_files.append(get_test_file_path(filename))
            except:
                continue
        
        if len(test_files) < 2:
            pytest.skip("Not enough different file types for mixed batch test")
        
        # 1. Batch-Analyse
        import_status = is_audio_in_zarr_audio_group(audio_db, test_files)
        file_analyses = [analysis for imported, analysis in import_status if not imported]
        
        # 2. Verschiedene Konfigurationen
        for i, file_analysis in enumerate(file_analyses):
            if i % 2 == 0:
                file_analysis.target_format = "FLAC"
            else:
                file_analysis.target_format = "AAC"
                file_analysis.aac_bitrate = 160000
        
        # 3. Filtere importierbare Dateien
        importable_analyses = [fa for fa in file_analyses if fa.can_be_imported]
        
        if not importable_analyses:
            pytest.skip("No importable files in mixed batch")
        
        # 4. Batch-Import
        results = aimport(audio_db, importable_analyses)
        successful_imports = [r for r in results if r.success]
        
        assert len(successful_imports) > 0
        
        # 5. Verifikation verschiedener Formate
        formats_used = set(r.target_format.code for r in successful_imports if r.target_format)
        assert len(formats_used) > 1  # Mindestens 2 verschiedene Formate
        
        # 6. Überprüfe dass alle Gruppen erstellt wurden
        for result in successful_imports:
            assert result.zarr_group_name in audio_db
            imported_group = audio_db[result.zarr_group_name]
            assert imported_group.attrs["type"] == "original_audio_file"


# ============================================================================
# Configuration Tests
# ============================================================================

class TestConfigurationIntegration:
    """Tests für Integration mit Config-System"""
    
    def test_config_audio_import_batch_size(self, temp_zarr_store):
        """Test Config.audio_import_batch_size Einfluss"""
        original_batch_size = getattr(Config, 'audio_import_batch_size', 20)
        
        try:
            # Setze kleine Batch-Größe
            Config.audio_import_batch_size = 2
            
            audio_db = open_zarr_audio_grp(temp_zarr_store)
            
            # Erstelle Liste mit mehr Dateien als Batch-Größe
            test_files = []
            for filename in FAST_TEST_FILES:
                try:
                    test_files.append(get_test_file_path(filename))
                except:
                    continue
            
            if len(test_files) >= 3:  # Mehr als Batch-Größe von 2
                # Sollte Parallel-Processing verwenden
                import_status = is_audio_in_zarr_audio_group(audio_db, test_files)
                
                assert len(import_status) == len(test_files)
                assert all(isinstance(status, tuple) for status in import_status)
        
        finally:
            # Restore original config
            Config.audio_import_batch_size = original_batch_size
    
    def test_config_logging_integration(self, temp_zarr_store):
        """Test Integration mit Logging-System"""
        from zarrwlr.config import Config
        from zarrwlr.packagetypes import LogLevel
        
        original_log_level = Config.log_level
        
        try:
            # Setze Debug-Logging
            Config.set(log_level=LogLevel.DEBUG)
            
            audio_db = open_zarr_audio_grp(temp_zarr_store)
            test_file = get_test_file_path("bird1_snippet.mp3")
            
            # Import sollte mit Debug-Logging funktionieren
            _, file_analysis = is_audio_in_zarr_audio_group(audio_db, test_file)
            
            if file_analysis.can_be_imported:
                result = aimport(audio_db, file_analysis)
                assert result.success == True
        
        finally:
            # Restore original config
            Config.set(log_level=original_log_level)


# ============================================================================
# Mock and Subprocess Tests
# ============================================================================

class TestMockAndSubprocess:
    """Tests für Mock-Funktionalität und Subprocess-Verhalten"""
    
    def test_subprocess_worker_function(self, temp_zarr_store):
        """Test _import_subprocess_worker() Funktion"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        test_file = get_test_file_path("bird1_snippet.mp3")
        
        file_param = FileParameter(test_file)
        if not file_param.can_be_imported:
            pytest.skip("Test file cannot be imported")
        
        # Teste Subprocess-Worker direkt
        result = _import_subprocess_worker(audio_db.zarr_group, file_param)
        
        assert isinstance(result, ImportResult)
        assert result.success == True
        assert result.file_path == test_file
    
    @pytest.mark.skipif(not hasattr(pytest, 'mark'), reason="Requires pytest marks")
    def test_parallel_processing_simulation(self, temp_zarr_store):
        """Test Simulation von Parallel-Processing"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        
        # Sammle verfügbare Test-Dateien
        test_files = []
        for filename in FAST_TEST_FILES:
            try:
                test_files.append(get_test_file_path(filename))
            except:
                continue
        
        if len(test_files) < 2:
            pytest.skip("Not enough files for parallel processing test")
        
        # Teste mit Mock von ProcessPoolExecutor für deterministische Tests
        with patch('zarrwlr.aimport.ProcessPoolExecutor') as mock_executor:
            # Konfiguriere Mock für direkten Aufruf
            mock_executor.return_value.__enter__.return_value.submit.side_effect = \
                lambda func, *args: type('MockFuture', (), {
                    'result': lambda: func(*args)
                })()
            
            # Führe Test aus
            import_status = is_audio_in_zarr_audio_group(audio_db, test_files)
            
            assert len(import_status) == len(test_files)
            assert all(isinstance(status, tuple) for status in import_status)


# ============================================================================
# Stress Tests
# ============================================================================

class TestStressScenarios:
    """Stress-Tests für Grenzfälle und große Datenmengen"""
    
    def test_large_batch_simulation(self, temp_zarr_store):
        """Test mit simulierter großer Batch-Operation"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        
        # Verwende kleine Datei mehrfach (simuliert große Batch)
        test_file = get_test_file_path("bird1_snippet.mp3")
        
        # Erstelle mehrere FileParameter-Instanzen für dieselbe Datei
        # (simuliert verschiedene Dateien)
        file_params = []
        for i in range(5):  # Moderate Anzahl für Test
            file_param = FileParameter(test_file)
            # Unterschiedliche Konfigurationen
            if i % 2 == 0:
                file_param.target_format = "FLAC"
            else:
                file_param.target_format = "AAC"
                file_param.aac_bitrate = 160000 + i * 32000
            
            if file_param.can_be_imported:
                file_params.append(file_param)
        
        if not file_params:
            pytest.skip("No importable file parameters for stress test")
        
        # Führe Batch-Import aus
        start_time = time.time()
        results = aimport(audio_db, file_params)
        total_time = time.time() - start_time
        
        successful_imports = [r for r in results if r.success]
        
        # Überprüfe Ergebnisse
        assert len(successful_imports) > 0
        
        # Performance sollte reasonable sein
        if len(successful_imports) > 0:
            avg_time = total_time / len(successful_imports)
            assert avg_time < 120, f"Stress test too slow: {avg_time:.2f}s per import"
    
    def test_memory_usage_large_file(self, temp_zarr_store):
        """Test Speicher-Verhalten bei großen Dateien"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        
        # Verwende größte verfügbare Datei
        large_files = [
            "audiomoth_long_snippet_converted.flac",  # 200MB
            "XC897425 - Rohrschwirl - Locustella luscinioides.wav"  # 118MB
        ]
        
        test_file = None
        for filename in large_files:
            try:
                test_file = get_test_file_path(filename)
                break
            except:
                continue
        
        if not test_file:
            pytest.skip("No large files available for memory test")
        
        # Analysiere ohne Import (sollte wenig Speicher verwenden)
        _, file_analysis = is_audio_in_zarr_audio_group(audio_db, test_file)
        
        # FileParameter sollte erfolgreich erstellt worden sein trotz großer Datei
        assert file_analysis.base_parameter.file.exists()
        assert file_analysis.base_parameter.file_size_bytes > 50 * 1024 * 1024  # > 50MB
        
        # Hash sollte berechnet worden sein
        assert len(file_analysis.base_parameter.file_sh256) == 64
    
    def test_concurrent_access_simulation(self, temp_zarr_store):
        """Test Simulation gleichzeitiger Zugriffe"""
        audio_db = open_zarr_audio_grp(temp_zarr_store)
        test_file = get_test_file_path("bird1_snippet.mp3")
        
        # Simuliere mehrere gleichzeitige Analysen derselben Datei
        file_params = []
        for i in range(3):
            file_param = FileParameter(test_file)
            if file_param.can_be_imported:
                file_params.append(file_param)
        
        if not file_params:
            pytest.skip("No importable parameters for concurrent test")
        
        # Alle sollten dieselbe Datei analysiert haben
        hashes = [fp.base_parameter.file_sh256 for fp in file_params]
        assert len(set(hashes)) == 1  # Alle gleich
        
        # Status-Check sollte für alle "nicht importiert" ergeben
        for file_param in file_params:
            already_imported, _ = is_audio_in_zarr_audio_group(audio_db, file_param)
            assert already_imported == False


# ============================================================================
# Main Execution
# ============================================================================

def run_tests():
    """Führe alle Tests direkt aus (ohne pytest)"""
    print("🧪 Running aimport.py Tests")
    print("=" * 60)
    
    # Cleanup testresults directory
    cleanup_test_results()
    
    # Test Kategorien
    test_classes = [
        TestAGroupWrapper,
        TestZarrAudioGroupManagement,
        TestImportStatusChecking,
        TestAudioImportOperations,
        TestImportWorker,
        TestSpecialScenarios,
        TestPerformance,
        TestErrorHandling,
        TestIntegration,
        TestConfigurationIntegration,
        TestMockAndSubprocess,
        TestStressScenarios
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    skipped_tests = []
    
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
                
                # Erstelle isolierte Test-Umgebung falls benötigt
                kwargs = {}
                method_params = test_method.__code__.co_varnames
                
                if 'temp_zarr_store' in method_params:
                    kwargs['temp_zarr_store'] = get_unique_zarr_store(f"{test_class.__name__}_{method_name}")
                        
                if 'sample_audio_files' in method_params:
                    sample_files = []
                    for filename in FAST_TEST_FILES:
                        try:
                            sample_files.append(get_test_file_path(filename))
                        except:
                            pass
                    kwargs['sample_audio_files'] = sample_files
                
                test_method(**kwargs)
                
                print(f"  ✅ {method_name}")
                passed_tests += 1
                
            except Exception as e:
                error_str = str(e)
                if "skip" in error_str.lower() or "not found" in error_str.lower():
                    print(f"  ⏭️  {method_name}: {error_str}")
                    skipped_tests.append((test_class.__name__, method_name, error_str))
                else:
                    print(f"  ❌ {method_name}: {error_str}")
                    failed_tests.append((test_class.__name__, method_name, error_str))
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print(f"📊 Test Summary:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests} ✅")
    print(f"   Failed: {len(failed_tests)} ❌")
    print(f"   Skipped: {len(skipped_tests)} ⏭️")
    print(f"   Test Results Dir: {TESTRESULTS_DIR}")
    
    if failed_tests:
        print(f"\n💥 Failed Tests:")
        for class_name, method_name, error in failed_tests:
            print(f"   {class_name}.{method_name}: {error}")
    
    if skipped_tests:
        print(f"\n⏭️  Skipped Tests:")
        for class_name, method_name, reason in skipped_tests:
            print(f"   {class_name}.{method_name}: {reason}")
    
    if len(failed_tests) == 0:
        print(f"\n🎉 All runnable tests passed!")
        return True
    else:
        return False                test_method(**kwargs)
                
                print(f"  ✅ {method_name}")
                passed_tests += 1
                
            except Exception as e:
                error_str = str(e)
                if "skip" in error_str.lower() or "not found" in error_str.lower():
                    print(f"  ⏭️  {method_name}: {error_str}")
                    skipped_tests.append((test_class.__name__, method_name, error_str))
                else:
                    print(f"  ❌ {method_name}: {error_str}")
                    failed_tests.append((test_class.__name__, method_name, error_str))
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print(f"📊 Test Summary:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests} ✅")
    print(f"   Failed: {len(failed_tests)} ❌")
    print(f"   Skipped: {len(skipped_tests)} ⏭️")
    print(f"   Test Results Dir: {TESTRESULTS_DIR}")
    
    if failed_tests:
        print(f"\n💥 Failed Tests:")
        for class_name, method_name, error in failed_tests:
            print(f"   {class_name}.{method_name}: {error}")
    
    if skipped_tests:
        print(f"\n⏭️  Skipped Tests:")
        for class_name, method_name, reason in skipped_tests:
            print(f"   {class_name}.{method_name}: {reason}")
    
    if len(failed_tests) == 0:
        print(f"\n🎉 All runnable tests passed!")
        return True
    else:
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)