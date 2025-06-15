# Umbau Status - Audio Storage System Refactoring
**Datum: 16.06.2025 23:00 | Status: Phase 2 von 4 abgeschlossen**

## 🎯 **Projekt-Überblick**

### **Grundsystem (Bestehend)**
- **Zarr v3 Backend**: High-Performance Audio-Speicherung mit Random Access auf kurze Sample-Ausschnitte
- **Dual-Codec Support**: FLAC (verlustfrei) und AAC-LC (verlustbehaftet) vollständig implementiert
- **Eigenes Index-System**: Für beide Kodierungen jeweils angepasste Indizierungsmethon implrmentiert, um fast Random Access zu gewährleisten
- **Performance**: AAC-Implementation ist **PRODUCTION READY** mit 50-80ms Extraktion, 1000x Speicher-Reduktion; FLAC ist ebenfalls uneingeschränkt **PRODUCTION READY**
- **Architektur**: Modulares System mit codec-spezifischen Access-Modulen

### **Umbau-Ziel**
Überarbeitung der **User-Level API** für bessere Nutzerfreundlichkeit:
- **Intelligente Parameter-Analyse** mit automatischen Codec- und Codec-Parameter-Vorschlägen
- **Strukturierter Workflow** über `FileParameter.analyze()` 
- **Qualitäts-erhaltende Suggestions** (nie Verschlechterung)
- **Konflikt-Detection** mit Nutzerberatung (Nutzerberatung fragt der Nutzer über die Print-Funktion der Klasse ab)
- **Einheitliche API** für beide Codecs (FLAC/AAC)

## 📋 **Aktueller Implementierungsstand**

### **✅ ABGESCHLOSSEN - Phase 1: Basis-Module erweitert**

#### **1. source_file.py (Komplett überarbeitet)**
**Status**: ✅ **FERTIG** - Kompletter Ersatz der ursprünglichen Datei

**Neue Kernklassen:**
- **`QualityAnalyzer`**: Intelligente Qualitätsanalyse und Parameter-Vorschläge
- **`ConflictAnalyzer`**: Erkennung von Import-Konflikten und Qualitätsproblemen
- **Enhanced `FileParameter`**: Erweitert um Target-Parameter-Management

**Neue FileParameter Properties:**
```python
# Auto-Trigger Properties (lösen _analyze() aus)
file_param.target_format = "AAC_44100"  # TargetFormats|str
file_param.target_sampling_transform = "EXACTLY"  # TargetSamplingTransforming|str  
file_param.aac_bitrate = 160000  # int
file_param.flac_compression_level = 4  # int

# Analyse-Ergebnisse
file_param.quality_analysis  # dict mit Qualitätsbewertung
file_param.conflicts  # dict mit blocking_conflicts, quality_warnings, efficiency_warnings
file_param.can_be_imported  # bool - berücksichtigt Konflikte
```

**Smart Suggestion System:**
- **User-Tracking**: `_user_defined_params` Set verfolgt explizit gesetzte Parameter
- **Intelligente Vorschläge**: Nur noch nicht user-definierte Parameter werden automatisch gesetzt
- **Qualitäts-Regeln**:
  - Lossy Source → AAC mit +10...25% Bitrate
  - Lossless Source → FLAC mit gleicher Sample-Rate
  - PCM Source → Qualitäts-abhängige Entscheidung

**Enhanced Print-Ausgabe (`__str__`):**
```
╭─ Audio File Analysis ────────────────────────────────╮
│ File: birds_morning_song.flac (47.3 MB)             │
│ Container: FLAC                                      │ 
├─ Audio Stream ──────────────────────────────────────┤
│ Codec: FLAC (lossless)                              │
│ Channels: 2 (stereo)                                │
│ Sample Rate: 96,000 Hz                              │
│ Duration: 5:23.7                                     │
├─ Recommended Import Settings ───────────────────────┤
│ ✅ Target: FLAC_96000 (preserve quality)            │
│ ✅ Sampling: EXACTLY (perfect match)                │
│ 🔧 Compression: Level 4 (balanced)                  │
│ ┌─ Rationale ──────────────────────────────────────┐ │
│ │ Source is lossless → keep lossless              │ │
│ │ High sample rate → preserve for analysis        │ │
│ └──────────────────────────────────────────────────┘ │
├─ Status ─────────────────────────────────────────────┤
│ 🟢 Ready for import                                  │
╰──────────────────────────────────────────────────────╯

Legend: ✅=User set, 🔧=Auto-suggested
```

**Konflikt-Detection:**
- **Blocking Conflicts**: 🚫 Verhindern Import (Sample-Rate-Inkompatibilität)
- **Quality Warnings**: ⚠️ Qualitätsverlust (Lossless→Lossy, Re-Encoding)
- **Efficiency Warnings**: 💡 Unnötiges Aufblähen (Low-Quality→FLAC)

#### **2. audio_coding.py (Erweitert)**
**Status**: ✅ **FERTIG** - Erweitert um flexible String-Konvertierung

**Neue Methoden:**
```python
# Flexibler String-Input (case-insensitive)
TargetFormats.from_string_or_enum("flac")       # → FLAC
TargetFormats.from_string_or_enum("AAC_44100")  # → AAC_44100  
TargetFormats.from_string_or_enum("flac44100")  # → FLAC_44100

TargetSamplingTransforming.from_string_or_enum("exactly")           # → EXACTLY
TargetSamplingTransforming.from_string_or_enum("resampling_48000")  # → RESAMPLING_48000
TargetSamplingTransforming.from_string_or_enum("resample")          # → RESAMPLING_NEAREST
```

**Unterstützte String-Formate:**
- **Case-insensitive**: `"flac"`, `"FLAC"`, `"Flac"`
- **Mit/ohne Underscore**: `"flac_44100"`, `"flac44100"`
- **Partial Matches**: `"resample"` → `RESAMPLING_NEAREST`
- **Kurze Formen**: `"exact"` → `EXACTLY`

### **📋 TODO - Phase 2: snd_import.py erstellen**

**Ziel**: Neue Haupt-Import-API die das `aimport.py` ersetzt

**Geplante API-Struktur:**

#### **Haupt-Import-Funktion:**
```python
def import_audio(source: str|pathlib.Path|FileParameter, 
                zarr_group: zarr.Group,
                timestamp: datetime.datetime = None,
                **kwargs) -> FileParameter:
    """
    Unified Import-Funktion:
    - Akzeptiert Dateipfad ODER vorbereiten FileParameter
    - Automatische Analyse wenn nur Pfad gegeben
    - Verwendet FileParameter.suggestions als Default
    - Überschreibbare Parameter via kwargs
    - Gibt erweiterten FileParameter zurück
    """
```

#### **Convenience Functions:**
Das muss noch genauer analysiert werden, was man hier benötigt.
```python
def quick_import(audio_file: str|pathlib.Path, 
                zarr_group: zarr.Group,
                timestamp: datetime.datetime = None) -> FileParameter:
    """Ein-Zeiler für Standard-Import mit Automatik"""

def analyze_audio(audio_file: str|pathlib.Path, 
                 target_format: str = None) -> FileParameter:
    """Nur Analyse ohne Import für Nutzer-Review"""

def batch_import(audio_files: list[str|pathlib.Path],
                zarr_group: zarr.Group,
                **kwargs) -> list[FileParameter]:
    """Batch-Import mit Progress-Feedback"""
```

#### **Workflow-Integration:**
```python
# Automatischer Workflow (für einfache Fälle)
result = quick_import("audio.wav", zarr_group)

# Analyse-gesteuerter Workflow (für Kontrolle)
file_param = analyze_audio("audio.wav")
print(file_param)  # Review suggestions
file_param.aac_bitrate = 192000  # Override if needed
result = import_audio(file_param, zarr_group)

# Direkter Workflow mit Overrides
result = import_audio("audio.wav", zarr_group, 
                     target_format="AAC_44100", 
                     aac_bitrate=160000)
```

### **📋 TODO - Phase 3: Integration & Testing**

#### **__init__.py Update:**
```python
# Hauptfunktionen direkt verfügbar machen
from .snd_import import (
    import_audio, quick_import, analyze_audio, batch_import
)
from .source_file import FileParameter
from .audio_coding import TargetFormats, TargetSamplingTransforming

__all__ = [
    'import_audio', 'quick_import', 'analyze_audio', 'batch_import',
    'FileParameter', 'TargetFormats', 'TargetSamplingTransforming'
]
```

#### **Legacy Compatibility (Optional):**
Legacy ist im derzeitigen Entwicklungsstand nicht erforderlich.

### **📋 TODO - Phase 4: Validierung & Dokumentation**

- **Test-Suite erweitern** für neue FileParameter-Features
- **Performance-Validation** der neuen API
- **Dokumentation aktualisieren** mit neuen Workflows
- **Beispiel-Scripts** für verschiedene Use-Cases

## 🎯 **Design-Prinzipien & Anforderungen**

### **Nutzer-Workflow Zielvorstellung:**
```python
# 1. Analyse-Phase
file_parameter = FileParameter("audio.wav")
print(file_parameter)  # Zeigt Streams + intelligente Vorschläge

# 2. Optional: Parameter anpassen  
file_parameter.target_format = "AAC_44100"  # Auto-trigger re-analysis
file_parameter.aac_bitrate = 192000         # Auto-trigger re-analysis

# 3. Import wenn ready
if file_parameter.can_be_imported:
    import_audio(file_parameter, zarr_group, timestamp)
else:
    print("Konflikte gefunden - siehe Vorschläge oben")
```

### **Qualitäts-Erhaltungs-Regeln:**
1. **Niemals Qualität verschlechtern** nicht, ohne explizite Nutzer-Entscheidung
2. **Lossy Sources**: AAC mit höherer Bitrate (+10-30%) vorschlagen
3. **Lossless Sources**: FLAC mit gleicher Sample-Rate bevorzugen
4. **Bit-Depth**: Nicht unnötig aufblähen (24bit→32bit vermeiden)
5. **Sample-Rate**: FLAC bis rund 655kHz (Maximalwert im Header beschränkt), darüber Reinterpretation vorschlagen
6. **Lossles zu Lossy vom Nutzer gewünscht**: Schlage bei über 48kHz bis 96KHz Resampling auf 48kHz vor. Darüber gehen wir von Ultraschallaufnahmen aus: schlage Reinterpretation auf 24KHz vor, damit die Filter nicht zu viel vom oberen Frequentband entfernen.

### **Konflikt-Handling:**
- **Auto-Detection**: Inkompatible Parameter automatisch erkennen
- **User-Guidance**: Konkrete Lösungsvorschläge in print()-Ausgabe
- **Blocking**: `can_be_imported=False` bei kritischen Konflikten
- **Warnings**: Qualitäts-/Effizienz-Warnungen ohne Import-Blockierung

### **String/Enum Flexibilität:**
- **Case-insensitive**: Alle Groß-/Kleinschreibungen akzeptieren
- **Flexible Formate**: Mit/ohne Unterstriche, Partial Matches
- **Hilfreiche Fehler**: Verfügbare Optionen bei ungültiger Eingabe anzeigen

## 🔧 **Technische Integration**

### **Bestehende Module (unverändert):**
- **`flac_access.py`**: ✅ Produktionsreif, keine Änderungen nötig
- **`aac_access.py`**: ✅ Produktionsreif, keine Änderungen nötig  
- **`flac_index_backend.py`**: ✅ Optimiert, keine Änderungen nötig
- **`aac_index_backend.py`**: ✅ Optimiert, keine Änderungen nötig
- **`config.py`**: ✅ AAC-Parameter bereits integriert
- **`utils.py`**: ✅ Alle benötigten Helpers vorhanden

### **Integration-Points:**
```python
# snd_import.py wird delegieren an:
if target_format.code == 'flac':
    from .flac_access import import_flac_to_zarr
    result = import_flac_to_zarr(zarr_group, audio_file, source_params, ...)
    
elif target_format.code == 'aac':
    from .aac_access import import_aac_to_zarr  
    result = import_aac_to_zarr(zarr_group, audio_file, source_params, ...)
```

### **Parameter-Mapping:**
```python
def _map_file_parameter_to_import_params(file_param: FileParameter) -> dict:
    """Convert FileParameter to codec-specific import parameters"""
    base_params = {
        'audio_file': file_param.base_parameter.file,
        'source_params': _extract_source_params(file_param),
        'temp_dir': '/tmp'
    }
    
    if file_param.target_format.code == 'flac':
        base_params['flac_compression_level'] = file_param.flac_compression_level
    elif file_param.target_format.code == 'aac':
        base_params['aac_bitrate'] = file_param.aac_bitrate
    
    return base_params
```

## 🚨 **Bekannte Implementierungs-Details**

### **FileParameter._analyze() Trigger-System:**
- **Auto-Trigger**: Jede Property-Änderung löst `_analyze()` aus
- **User-Tracking**: `_user_defined_params` Set verhindert Override von User-Choices
- **Re-Suggestion**: Nur nicht-user-definierte Parameter werden neu vorgeschlagen

### **Conflict Detection Logic:**
```python
# ConflictAnalyzer.analyze_conflicts() gibt zurück:
{
    'blocking_conflicts': [],     # Import-verhindernde Probleme
    'quality_warnings': [],      # Qualitätsverlust-Warnungen  
    'efficiency_warnings': []    # Aufbläh-Warnungen
}
```

### **Print-Ausgabe Formatierung:**
- **Unicode Box-Drawing**: ╭─╮├─┤╰─╯ für saubere Darstellung
- **Status-Icons**: ✅🔧🚫⚠️💡 für schnelle visuelle Erfassung
- **Text-Wrapping**: `_wrap_text()` für lange Konflik-Beschreibungen
- **Legend**: Erklärung der Icons am Ende der Ausgabe

## 📂 **Datei-Status Übersicht**

```
Projekt-Root/
├── source_file.py          ✅ ERSETZT - Komplett neue Version implementiert
├── audio_coding.py         ✅ ERWEITERT - from_string_or_enum() hinzugefügt
├── snd_import.py           📋 TODO - Neue Haupt-API erstellen
├── __init__.py             📋 TODO - Imports für neue API aktualisieren
├── aimport.py              🗑️ ERSETZEN - Wird durch snd_import.py ersetzt
├── config.py               ✅ FERTIG - AAC-Parameter bereits vorhanden
├── flac_access.py          ✅ FERTIG - Keine Änderungen nötig
├── aac_access.py           ✅ FERTIG - Keine Änderungen nötig
├── flac_index_backend.py   ✅ FERTIG - Keine Änderungen nötig
├── aac_index_backend.py    ✅ FERTIG - Keine Änderungen nötig
├── utils.py                ✅ FERTIG - Alle Helper vorhanden
├── exceptions.py           ✅ FERTIG - Keine Änderungen nötig
├── packagetypes.py         ✅ FERTIG - Keine Änderungen nötig
└── logsetup.py             ✅ FERTIG - Keine Änderungen nötig
```

## 🎯 **Nächste konkrete Schritte**

### **Unmittelbar (nächster Chat):**
1. **`snd_import.py` erstellen** mit den geplanten Haupt-Funktionen
2. **Integration testen** - FileParameter → Import-Pipeline
3. **Parameter-Mapping** implementieren (FileParameter → Codec-Module)

### **Validierung:**
1. **Test-Script** für den kompletten Workflow schreiben
2. **Edge-Cases** testen (Konflikte, ungültige Parameter)
3. **Performance** der neuen API validieren

### **Finalisierung:**
1. **`__init__.py`** für neue API aktualisieren
2. **Legacy-Wrapper** optional erstellen
3. **Dokumentation** mit Beispielen aktualisieren

## 💡 **Design-Entscheidungen Getroffen**

1. **Auto-Trigger**: Property-Änderungen triggern sofort Re-Analyse (vs. expliziter analyze()-Aufruf)
2. **Direct Parameter Storage**: Suggestions werden direkt als Parameter gespeichert (vs. separate suggestion-Properties)
3. **Smart Override**: User-definierte Parameter werden nie überschrieben
4. **Complete File Replacement**: Komplette source_file.py ersetzt (vs. inkrementelle Patches)
5. **Unicode Box Output**: Schöne Terminal-Ausgabe mit Box-Drawing-Zeichen
6. **Three-Tier Conflicts**: blocking/quality/efficiency Kategorien für Nutzer-Guidance

**Projekt ist bereit für Phase 2: snd_import.py Implementation!**