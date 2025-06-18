# aimport.py Umbau - Implementation Plan

## Context
Vollständiger Umbau von `aimport.py` zu einer High-Level User API für Audio-Import in Zarr-Datenbanken. Keine Rückwärtskompatibilität erforderlich.

## Key Design Principles
- **High-Level User API**: Wenige, einfache Funktionen für Nicht-Python-Experten
- **Single/List Flexibility**: Jede Funktion akzeptiert einzelne Parameter oder Listen
- **Parallel Processing**: Listen werden in Subprozessen verarbeitet (eigene GIL)
- **Performance Optimization**: Gruppierung bei vielen kleinen Operationen (Config-Parameter)
- **FileParameter Integration**: Nutzt import_utils.FileParameter für Analyse und Konfiguration

## Implementation Details

### 1. AGroup Type Class
```python
class AGroup:
    """Simple wrapper for zarr.Group to enable isinstance() checks"""
    def __init__(self, zarr_group: zarr.Group):
        self._group = zarr_group
    
    # Delegate all zarr.Group methods
```
- Einfacher Wrapper für `isinstance()` checks
- Direkt in `aimport.py` definiert
- Alle zarr.Group Methoden durchreichen

### 2. Core Functions

#### open_zarr_audio_grp()
```python
def open_zarr_audio_grp(store_path: str|pathlib.Path, 
                       group_path: str|pathlib.Path|None = None, 
                       create: bool = True) -> AGroup:
```
- Verwendet `init_zarr_audio_grp()` und `check_if_zarr_audio_grp()`
- Mit `create=False`: Exception wenn nicht existent
- Automatisches Upgrade via `upgrade_zarr_audio_grp()` falls nötig
- Gibt `AGroup` zurück

#### is_audio_in_zarr_audio_group()
```python
def is_audio_in_zarr_audio_group(
    zarr_audio_group: AGroup,
    files: str|pathlib.Path|FileParameter|list[str]|list[pathlib.Path]|list[FileParameter]
) -> tuple[bool, FileParameter]|list[tuple[bool, FileParameter]]:
```
- Parallel processing für Listen mit Gruppierung (Config-Parameter für Gruppengröße)
- Auto-FileParameter Erstellung wenn nur Paths übergeben
- Gruppierung: Max 20 Files pro Prozess (Config-Parameter)

#### aimport()
```python
def aimport(zarr_audio_group: AGroup, 
           file_params: FileParameter|list[FileParameter]) -> ImportResult|list[ImportResult]:
```
- **Keine `**kwargs`** - alle Parameter über FileParameter
- Vorbedingung: Prüfung aller `file_params.can_be_imported` BEVOR parallel processing
- Exception wenn ANY blocking conflicts
- Worker-Klasse für direct/subprocess execution

### 3. Supporting Functions

#### Zarr Management
- `init_zarr_audio_grp()` - rename von `init_original_audio_group()`
- `check_if_zarr_audio_grp()` - return bool instead of exceptions
- `upgrade_zarr_audio_grp()` - empty template returning True

#### Worker Architecture
```python
class ImportWorker:
    def run_direct(self, file_param: FileParameter) -> ImportResult:
        # Direct execution in main process
    
    def run_subprocess(self, file_param: FileParameter) -> ImportResult:
        # Subprocess execution with pipes
```

### 4. Performance Optimizations
- **Gruppierung bei Lists**: Nicht jede Operation einzeln an Subprozess
- **Config-Parameter**: `audio_import_batch_size = 20` für is_audio_in_zarr_audio_group
- **Import bleibt File-Level**: Große Audio-Files rechtfertigen einen Prozess pro File

### 5. Removed Functions
Diese entfallen komplett:
- `import_original_audio_file_with_config`
- `validate_aac_import_parameters`
- `_get_aac_config_for_import`
- `_log_import_performance`
- `test_aac_integration`
- `_get_source_params`
- `safe_get_sample_format_dtype`

### 6. Module Documentation
- **Ausführlicher Nutzer-Docstring** ähnlich `import_utils.py`
- **Einfache Sprache** für Nicht-Python-Experten
- **Vollständige Beispiele** für typische Use Cases
- **FileParameter Integration** erklären
- **Funktionsübergreifende Workflows** dokumentieren

### 7. Typical User Workflow
```python
# 1. Open/create Zarr audio group
agroup = open_zarr_audio_grp("./my_audio_db")

# 2. Check if files already imported
was_imported, file_params = is_audio_in_zarr_audio_group(agroup, "my_audio.wav")

# 3. Review analysis and configure if needed
if not was_imported:
    print(file_params)  # Beautiful terminal output
    # Optional: file_params.target_format = "AAC_44100"

# 4. Import with intelligent suggestions
result = aimport(agroup, file_params)
```

### 8. Config Extensions Needed
Neuer Parameter in `config.py`:
```python
audio_import_batch_size: int = 20  # Files per subprocess for batch operations
```

### 9. Import Requirements
- `from .import_utils import FileParameter`
- Existing FLAC/AAC access modules
- `from .config import Config`
- Standard libraries für parallel processing

## Next Steps
1. Create complete `aimport.py` with full module docstring
2. Test basic functionality
3. Add Config parameter if needed
4. Review and optimize parallel processing logic