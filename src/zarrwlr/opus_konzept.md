# OPUS-Parallelisierung Implementierungskonzept

## Projektkontext und Zielsetzung

### Überblick
Basierend auf der erfolgreichen FLAC-Parallelisierung soll eine nahezu identische Implementierung für OGG-Opus erstellt werden. Das Ziel ist eine hochperformante, parallelisierte Opus-Index-Erstellung mit denselben Architekturprinzipien wie bei FLAC.

### Projektstruktur
```
./src/zarrwlr/
├── aimport.py                    # Import-Orchestrator (zu bereinigen)
├── flac_access.py               # FLAC Public API (Template)
├── flac_index_backend.py        # FLAC Parallelisierung (Template)
├── opus_access.py               # ZU ERSTELLEN - Opus Public API
├── opus_index_backend.py        # ZU ERSTELLEN - Opus Parallelisierung
├── opusbyteblob.py             # LEGACY - zu ersetzen
└── ...

./tests/
├── test_flac_*.py              # FLAC Tests (Template)
├── test_opus_*.py              # ZU ERSTELLEN - Opus Tests
└── testdata/
    ├── audiomoth_long_snippet.wav
    ├── audiomoth_short_snippet.wav
    └── bird1_snippet.mp3
```

## Aktuelle Situation

### ✅ Erfolgreich implementiert (FLAC):
- **3-Phasen Parallelisierung**: Sync-Suche → Frame-Details → Sample-Akkumulation
- **Memory-effiziente Zarr-Referenzen**: Keine Daten-Kopierung zwischen Prozessen
- **Robuste API**: `flac_access.py` + `flac_index_backend.py`
- **Performance**: ~3-5x Speedup gegenüber Sequential
- **Vollständige Tests**: Korrektheit, Performance, Integrity

### ❌ Zu ersetzen (Opus):
- **Alte Sequential-Implementation**: `opusbyteblob.py`
- **Monolithische Struktur**: Alles in einer Datei
- **Keine Parallelisierung**: Langsame Index-Erstellung
- **Legacy-Code in aimport.py**: Nicht wartbar, nicht erweiterbar

## Benötigte Dateien für vollständigen Kontext

### Bereits vorhanden (als Template):
1. **`./src/zarrwlr/flac_access.py`** - Public API Template
2. **`./src/zarrwlr/flac_index_backend.py`** - Parallelisierung Template
3. **`./tests/test_flac_*.py`** - Test-Template

### Aktueller Stand (zu analysieren):
4. **`./src/zarrwlr/aimport.py`** - Import-Orchestrator mit Opus-Code
5. **`./src/zarrwlr/opusbyteblob.py`** - Legacy Opus-Implementation

### Testdaten:
6. **`./tests/testdata/`** - Audio-Testdateien für Validierung

### Bei Chat-Start zu fragen:
- "Benötige die 5 oben genannten Dateien für vollständigen Kontext"
- Falls neue Testdateien nötig: "Sind Opus-spezifische Testdateien verfügbar?"

## Opus-spezifische Besonderheiten

### 1. **1:1 Datenübernahme bei Opus-Quelle**
```python
# In opus_access.py
if source_codec == "opus" and target_codec == "opus":
    # Keine Umkodierung - direkte Übernahme der OGG-Container-Daten
    # Vorteil: Perfekte Qualität, keine Verluste
    copy_opus_data_directly_to_zarr()
```

### 2. **Ultrasonic-Handling (>48kHz)**
```python
# Opus-Limitation: Max 48kHz
if source_sample_rate > 48000:
    # Trick: Daten als 48kHz interpretieren, Faktor speichern
    sampling_rescale_factor = source_sample_rate / 48000.0
    target_sample_rate = 48000
    # Bei Extraktion: Metadaten mit ursprünglicher Rate zurückgeben
    metadata['original_sample_rate'] = source_sample_rate
```

### 3. **OGG-Container-Struktur vs. FLAC**
| Aspekt | FLAC | Opus (OGG) |
|--------|------|------------|
| Container | Native FLAC | OGG Pages |
| Sync-Pattern | 0xFFF8 | "OggS" |
| Frame-Größe | Relativ konstant | Variable Page-Größe |
| Sample-Position | Akkumuliert | Granule-Position |
| Header-Komplexität | Einfach | Segment-Tabellen |

## Zielarchitektur

### API-Kompatibilität (identisch zu FLAC):
```python
# opus_access.py - Identische Schnittstelle wie FLAC
def import_opus_to_zarr(zarr_group, audio_file, source_params, 
                       first_sample_time_stamp, opus_bitrate=160000, temp_dir="/tmp"):
    """Import audio to Opus format with automatic indexing"""
    
def extract_audio_segment_opus(zarr_group, audio_blob_array, start_sample, end_sample, dtype=np.int16):
    """Extract audio segment from indexed Opus database"""
    
def parallel_extract_audio_segments_opus(zarr_group, audio_blob_array, segments, dtype=np.int16, max_workers=4):
    """Parallel extraction of multiple segments"""
    
def build_opus_index(zarr_group, audio_blob_array, use_parallel=True, max_workers=None):
    """Create index for Opus frame access with parallelization"""
```

## 3-Phasen Parallelisierung für Opus

### **Phase 1: Parallel OGG-Page-Suche**
```python
# opus_index_backend.py
def _find_ogg_pages_parallel(zarr_store_path, group_path, array_name, max_workers):
    """
    Memory-effiziente parallele Suche nach OGG-Page-Headern
    
    Unterschiede zu FLAC:
    - Suche nach b'OggS' statt 0xFFF8
    - Variable Page-Größen (FLAC: relativ konstante Frame-Größen)
    - Komplexere Header-Struktur mit Segment-Tabellen
    
    Returns:
        Sortierte Liste aller OGG-Page-Positionen
    """
```

### **Phase 2: Parallel Page-Detail-Berechnung**
```python
def _process_ogg_pages_parallel(zarr_store_path, group_path, array_name, 
                               page_positions, max_workers):
    """
    Für jede Page parallel:
    - Page-Größe berechnen (Header + Segment-Tabelle + Body)
    - Granule-Position extrahieren (64-bit Sample-Position)
    - Page-Hash für Validierung
    - Segment-Tabelle parsen
    
    OGG-spezifische Berechnungen:
    - Segment-Tabelle: Anzahl und Größe der Segmente
    - Page-Body-Größe: Summe aller Segment-Größen
    - Granule-Position: Absolute Sample-Position im Stream
    
    Returns:
        Vollständige Page-Details (ohne Sample-Positionen)
    """
```

### **Phase 3: Sequential Sample-Position-Akkumulation**
```python
def _accumulate_sample_positions_opus(page_details):
    """
    Akkumuliere Sample-Positionen basierend auf Granule-Positionen
    
    Opus-spezifische Korrekturen:
    - Granule-Interpolation bei fehlenden Granule-Positionen
    - Ultrasonic-Korrektur: sampling_rescale_factor anwenden
    - Kontinuität sicherstellen
    
    Returns:
        Finaler Opus-Index mit korrigierten Sample-Positionen
    """
```

### **Index-Array-Format**
```python
# Analog zu FLAC_INDEX_*
OPUS_INDEX_DTYPE = np.uint64
OPUS_INDEX_COLS = 3  # [byte_offset, page_size, sample_pos]
OPUS_INDEX_COL_BYTE_OFFSET = 0
OPUS_INDEX_COL_PAGE_SIZE = 1  
OPUS_INDEX_COL_SAMPLE_POS = 2
```

## Implementierungsfahrplan

### **Phase 1: Grundlegende Struktur (1-2 Tage)**

#### **Schritt 1.1: opus_access.py erstellen**
- [ ] **Template aus flac_access.py kopieren**
- [ ] **Opus-spezifische Parameter anpassen**
- [ ] **import_opus_to_zarr() implementieren**
  ```python
  def import_opus_to_zarr(zarr_group, audio_file, source_params, 
                         first_sample_time_stamp, opus_bitrate=160000, temp_dir="/tmp"):
      # 1:1 Kopie für Opus-Quellen (source_params["is_opus"] == True)
      if source_params["is_opus"] and not is_ultrasonic:
          return copy_opus_data_directly()
      
      # Ultrasonic-Handling (>48kHz)
      if source_sample_rate > 48000:
          sampling_rescale_factor = source_sample_rate / 48000.0
          ffmpeg_cmd += ["-ar", "48000"]
      
      # Standard ffmpeg-Encoding für andere Codecs
      return encode_to_opus_via_ffmpeg()
  ```
- [ ] **extract_audio_segment_opus() implementieren**
  - [ ] Temporäre OGG-Datei-Erstellung
  - [ ] ffmpeg-basierte Dekodierung  
  - [ ] Sample-Rate-Korrektur für Ultrasonic

#### **Schritt 1.2: Basis opus_index_backend.py**
- [ ] **Sequential Fallback implementieren** (aus opusbyteblob.py portieren)
  ```python
  def _parse_ogg_pages_sequential(audio_bytes, expected_sample_rate=48000):
      """
      Sequential OGG-Page-Parsing (Fallback-Implementation)
      Portiert aus opusbyteblob.py aber strukturiert wie FLAC
      """
  ```
- [ ] **build_opus_index() Haupt-API**
  ```python
  def build_opus_index(zarr_group, audio_blob_array, use_parallel=True, max_workers=None):
      """Analog zu build_flac_index() aber für Opus"""
  ```

#### **Schritt 1.3: Integration in aimport.py**
- [ ] **Opus-Code aus aimport.py in opus_access.py verschieben**
- [ ] **Neue opus_access API einbinden**
- [ ] **Orchestrator-Logik vereinfachen**
  ```python
  # aimport.py - Vereinfachte Struktur
  if target_codec == 'flac':
      return flac_access.import_flac_to_zarr(...)
  elif target_codec == 'opus':
      return opus_access.import_opus_to_zarr(...)
  ```

**Test nach Phase 1:**
```bash
cd tests
python test_opus_basic.py  # Grundfunktionalität ohne Parallelisierung
```

### **Phase 2: Parallelisierung Kern (2-3 Tage)**

#### **Schritt 2.1: Phase 1 - OGG-Page-Suche**
- [ ] **Chunk-basierte OGG-Verarbeitung**
  ```python
  class OggChunkReference:
      """Analog zu ChunkReference aber für OGG-Pages"""
      def __init__(self, zarr_store_path, group_path, array_name,
                   start_byte, end_byte, chunk_id):
  
  def _find_ogg_pages_in_chunk(chunk_ref):
      """Worker-Funktion: Finde OGG-Pages in einem Chunk"""
      # Suche nach b'OggS' Signaturen
      # Memory-effiziente Zarr-Referenzen wie bei FLAC
  ```

#### **Schritt 2.2: Phase 2 - Page-Detail-Berechnung**
- [ ] **PageDetail-Klasse erstellen**
  ```python
  class PageDetail:
      """Analog zu FrameDetail aber für OGG-Pages"""
      def __init__(self, page_index, byte_offset, page_size, 
                   granule_position, page_hash):
          self.page_index = page_index
          self.byte_offset = byte_offset
          self.page_size = page_size
          self.granule_position = granule_position  # Statt estimated_samples
          self.page_hash = page_hash
          self.sample_position = None  # Wird in Phase 3 gesetzt
  ```
- [ ] **_process_single_ogg_page() implementieren**
  ```python
  def _process_single_ogg_page(page_ref):
      """
      Verarbeite eine einzelne OGG-Page parallel
      - OGG-Header-Parsing (27 Bytes + Segment-Tabelle)
      - Granule-Position-Extraktion (64-bit)
      - Page-Größe-Berechnung
      """
  ```

#### **Schritt 2.3: Phase 3 - Sample-Akkumulation**
- [ ] **_accumulate_sample_positions_opus() implementieren**
  ```python
  def _accumulate_sample_positions_opus(page_details, sampling_rescale_factor=1.0):
      """
      Opus-spezifische Sample-Akkumulation:
      - Granule-Position-basierte Berechnung
      - Ultrasonic-Faktor-Korrektur anwenden
      - Interpolation bei fehlenden Granule-Positionen
      """
  ```

**Test nach Phase 2:**
```bash
python test_opus_parallel_basic.py  # Parallelisierung ohne Edge Cases
```

### **Phase 3: Robustheit und Optimierung (2-3 Tage)**

#### **Schritt 3.1: Error Handling**
- [ ] **Korrupte OGG-Pages behandeln**
  ```python
  def _validate_ogg_page(page_data):
      """Validiere OGG-Page-Struktur"""
      # CRC-Check, Segment-Tabellen-Konsistenz, etc.
  ```
- [ ] **Graceful Fallback auf Sequential**
- [ ] **Incomplete Page-Daten abfangen**

#### **Schritt 3.2: Ultrasonic-Features**
- [ ] **sampling_rescale_factor Integration**
  ```python
  def _apply_ultrasonic_correction(sample_positions, rescale_factor):
      """Korrigiere Sample-Positionen für Ultrasonic-Dateien"""
      return sample_positions * rescale_factor
  ```
- [ ] **Metadaten-Korrektur bei Extraktion**
- [ ] **Edge-Case-Handling für sehr hohe Sample-Raten**

#### **Schritt 3.3: Performance-Optimierung**
- [ ] **Chunk-Größen für OGG optimieren** (anders als FLAC)
- [ ] **Worker-Anzahl konfigurierbar machen**
- [ ] **Memory-Usage minimieren**

**Test nach Phase 3:**
```bash
python test_opus_parallel_complete.py  # Vollständige Parallelisierung
```

### **Phase 4: Testing und Validierung (2-3 Tage)**

#### **Schritt 4.1: Performance-Tests**
- [ ] **test_opus_parallel.py erstellen** (Template: test_flac_parallel.py)
  - [ ] Parallel vs Sequential Correctness
  - [ ] Performance-Skalierung  
  - [ ] Memory-Effizienz
  - [ ] API-Konsistenz

#### **Schritt 4.2: Integrity-Tests**
- [ ] **test_opus_integrity.py erstellen** (Template: test_flac_integrity.py)
  - [ ] End-to-End Validierung
  - [ ] Original vs Database Sample-Vergleich
  - [ ] **Ultrasonic-Pipeline-Test** (Opus-spezifisch)
  - [ ] **1:1 Opus-Copy-Test** (Opus-spezifisch)

#### **Schritt 4.3: Production-Tests**
- [ ] **test_opus_production.py erstellen** (Template: test_flac_production.py)
  - [ ] CI/CD-kompatible Tests
  - [ ] Professional output (ohne Emojis)
  - [ ] Automated regression testing

**Test nach Phase 4:**
```bash
python test_opus_production.py --ci      # CI/CD ready
python test_opus_integrity.py --verbose  # End-to-end validation
```

### **Phase 5: Integration und Cleanup (1-2 Tage)**

#### **Schritt 5.1: aimport.py Bereinigung**
- [ ] **Komplette Entfernung von Opus-Code aus aimport.py**
- [ ] **Orchestrator-Logik vereinfachen**
- [ ] **Einheitliche API für FLAC/Opus**
- [ ] **opusbyteblob.py entfernen** (nach erfolgreicher Migration)

#### **Schritt 5.2: Final Testing**
- [ ] **Komplette Test-Suite für Opus ausführen**
- [ ] **Regressions-Tests für FLAC** (sicherstellen dass nichts kaputt ging)
- [ ] **Integration-Tests für beide Codecs**

## Debugging-Strategie

### **Incrementelle Entwicklung:**
1. **Sequential vor Parallel** implementieren
2. **Jeder Schritt einzeln testen** mit kleinen Dateien
3. **Ausführliches Logging** in Debug-Phase
4. **Isolierte Komponenten-Tests**

### **Test-Daten-Progression:**
1. **Kleine Mono-Dateien** (48kHz, Standard-Opus)
2. **Stereo-Dateien** (48kHz)  
3. **Ultrasonic-Dateien** (>48kHz) - Opus-spezifisch
4. **1:1 Opus-Übernahme** - Opus-spezifisch
5. **Große Dateien** (Performance-Test)

### **Isolierte Tests:**
```python
# Beispiel: Test nur OGG-Page-Parsing  
def test_single_ogg_page_parsing():
    # Lade kleine OGG-Datei
    # Parse erste Page
    # Validiere Header-Felder, Granule-Position, Segment-Tabelle

# Test nur Ultrasonic-Handling
def test_ultrasonic_rescaling():
    # 96kHz Input → 48kHz Opus mit Faktor 2.0
    # Validiere Metadaten-Korrektur
```

## Success Metrics

### **Performance-Ziele:**
- **3-5x Speedup** gegenüber Sequential (analog FLAC)
- **Memory-Effizienz**: <100MB RAM für große Dateien
- **Skalierbarkeit**: Linear mit Worker-Anzahl

### **Qualitäts-Ziele:**
- **100% Korrektheit**: Parallel == Sequential
- **Ultrasonic-Präzision**: Korrekte Sample-Rate-Korrektur
- **1:1 Opus-Copy**: Bit-genaue Übernahme ohne Umkodierung
- **Robustheit**: Graceful Fallback bei Fehlern

### **Integration-Ziele:**
- **API-Kompatibilität**: Identische Schnittstelle wie FLAC
- **aimport.py Vereinfachung**: Reiner Orchestrator ohne Codec-spezifischen Code
- **Test-Coverage**: 100% für kritische Pfade

## Bekannte Herausforderungen

### **OGG-spezifische Komplexität:**
- **Variable Page-Größen** erschweren Chunk-Aufteilung
- **Segment-Tabellen** komplexer als FLAC-Header
- **Granule-Position-Gaps** erfordern Interpolation

### **Ultrasonic-Handling:**
- **Metadaten-Konsistenz** zwischen Import und Extraktion
- **Edge-Cases** bei extremen Sample-Raten (>192kHz)
- **Performance-Impact** der Sample-Rate-Korrektur

### **1:1 Opus-Copy:**
- **Container-Validierung** ohne Dekodierung
- **Indexing ohne Re-Encoding**
- **Qualitätssicherung** bei direkter Übernahme

## Nächste Schritte für neuen Chat

### **Kontext-Anfrage:**
"Ich benötige für die Opus-Parallelisierung folgende 5 Dateien als Upload:
1. `./src/zarrwlr/flac_access.py` (Template)
2. `./src/zarrwlr/flac_index_backend.py` (Template)  
3. `./src/zarrwlr/aimport.py` (Opus-Code zu migrieren)
4. `./src/zarrwlr/opusbyteblob.py` (Legacy zu ersetzen)
5. Ein FLAC-Test als Template (z.B. `./tests/test_flac_integrity.py`)

Optional: Sind Opus-spezifische Testdateien verfügbar?"

### **Erste Entwicklungsaufgabe:**
**Schritt 1.1**: `opus_access.py` erstellen basierend auf `flac_access.py` Template mit Opus-spezifischen Anpassungen:
- `import_opus_to_zarr()` mit 1:1-Copy und Ultrasonic-Handling
- `extract_audio_segment_opus()` mit Sample-Rate-Korrektur
- Grundstruktur analog FLAC aber OGG-Container-spezifisch

**Mit diesem strukturierten Konzept und klaren Templates können wir systematisch die Opus-Parallelisierung implementieren und dabei die bewährte FLAC-Architektur als Basis nutzen.**