# AAC PROJECT CONCEPT - Random Access Implementation
**Date: 07.06.2025 | Status: CONCEPT DESIGN**

## 🎯 **Project Overview**

### **Mission Statement**
Implement a high-performance AAC-LC based audio storage and random access system using **Zarr v3** as the storage backend. The system shall import large audio files of arbitrary formats via **ffmpeg** into Zarr arrays as AAC-LC byte-streams, with comprehensive indexing to enable ultra-fast random access to small sample ranges without reading extensive portions of the AAC-LC stream.

**Core Requirements:**
- ✅ **Storage Backend:** Zarr v3 for scalable, chunked audio storage
- ✅ **Import Pipeline:** ffmpeg for universal audio format conversion to AAC-LC
- ✅ **Target Compression:** ~160 kbps (stereo) / ~130 kbps (mono) for optimal quality/size balance
- ✅ **Random Access:** Index-based extraction without ffmpeg subprocess overhead
- ✅ **Performance Goal:** Significantly better compression than FLAC with comparable random access speed
- ✅ **ffmpeg Usage:** Import-only (no random access usage due to subprocess overhead)

### **Technical Foundation**
The system leverages AAC-LC's **frame-based architecture** where each frame (~21ms audio) can be decoded independently after proper stream initialization. By creating a comprehensive **frame-level index**, we achieve true random access without the complex decoder state management required by stateful codecs like Opus.

**Architecture Principles:**
- **Zarr v3 Arrays:** Store AAC-LC byte-streams as chunked binary data with metadata
- **Frame Index Arrays:** Separate Zarr arrays containing frame positions, sizes, and timing
- **PyAV Decoding:** Native Python AAC decoding for zero subprocess overhead
- **Streaming Import:** Process large files efficiently during ffmpeg conversion
- **Sample-Accurate Access:** Frame-level granularity with sample-precise trimming

### **Key Advantages of AAC-LC Architecture**
- ✅ **Superior Compression:** 160kbps (stereo) / 130kbps (mono) vs FLAC ~650kbps
- ✅ **Frame Independence:** Each AAC frame can be decoded without complex state management
- ✅ **Zarr v3 Integration:** Leverages advanced chunking and metadata capabilities
- ✅ **Universal Input Support:** ffmpeg handles all audio formats during import
- ✅ **Minimal Decoder State:** ~1KB vs 18KB (Opus) = 18x smaller memory footprint
- ✅ **Native Python Processing:** PyAV eliminates subprocess overhead for random access
- ✅ **Production Proven:** 20+ years of optimization in professional audio workflows

## 📊 **Performance Targets**

### **Storage Efficiency Goals**
```
Target Compression (7+ minutes audio):
├── Original WAV: 20.4 MB (baseline)
├── FLAC Lossless: ~13 MB (36% reduction)
├── AAC-LC 160kbps: 8.6 MB (58% reduction vs WAV, 34% vs FLAC)
├── AAC + Index: 8.8 MB (57% reduction vs WAV, 32% vs FLAC)
└── Index Overhead: 0.2 MB (2.3% of compressed size)

Bitrate Targets:
├── Stereo Audio: ~160 kbps (optimal quality/size balance)
├── Mono Audio: ~130 kbps (equivalent quality density)
└── Quality Level: Transparent for most content at target bitrates
```

### **Performance Benchmarks**
```
Random Access Speed Targets:
├── FLAC Sequential: ~200-500ms (frame-by-frame decode)
├── AAC Frame-Access: ~15-25ms (40-65x speedup vs sequential)
├── vs ffmpeg subprocess: ~250ms (10x faster using PyAV)
└── Target Granularity: ~21ms (AAC frame duration)

Import Performance Goals:
├── Real-time Factor: >1.0x (faster than playback speed)
├── Memory Usage: <500MB peak during conversion
├── Large File Support: Multi-GB files via streaming
└── ffmpeg Integration: Single-pass conversion with index creation
```

## 🏗️ **System Architecture**

### **Core Components**

#### **1. AAC Import Pipeline**
```
Universal Audio Input → ffmpeg → AAC-LC Stream → Zarr v3 Storage → Frame Indexing
├── Input: Any format supported by ffmpeg (WAV, MP3, FLAC, OGG, etc.)
├── Conversion: ffmpeg AAC-LC encoding with precise bitrate control
├── Storage: Zarr v3 binary arrays with optimized chunking strategy
├── Indexing: Real-time frame analysis during import process
└── Output: Indexed AAC-LC stream in Zarr v3 format ready for random access

ffmpeg Integration:
├── Single-Use: Import-only, no random access operations
├── Streaming: Process large files without full memory loading
├── Quality Control: Precise bitrate and profile configuration
└── Format Independence: Universal audio format support
```

#### **2. Frame-Level Index Structure**
```
AAC Frame Index Format (per frame):
├── byte_offset: Position in AAC stream (uint64)
├── frame_size: Size in bytes (uint32) 
├── sample_count: Samples in frame (uint16, typically 1024)
├── sample_position: Cumulative sample position (uint64)
├── timestamp_ms: Time position in milliseconds (uint32)
└── frame_type: Regular/IDR frame marker (uint8)

Overhead Calculation:
├── Index Entry Size: 25 bytes per frame
├── Frames per Second: ~47 frames/sec (21.3ms per frame)
├── 7 Minutes Audio: ~19,740 frames
└── Total Index Size: ~494 KB (0.5 MB)
```

#### **3. Random Access System**
```
Random Access Flow:
├── Time/Sample Request → Binary Search Index → Frame Location
├── Seek to Frame Position → PyAV Decode → Extract Samples
├── Trim to Exact Range → Return Audio Data
└── No State Management Required (stateless frames)
```

## 🔧 **Technical Implementation Strategy**

### **Phase 1: AAC Import Module (`aac_access.py`)**

#### **Core Functions API**
```python
# Primary import function
def import_aac_to_zarr(zarr_group: zarr.Group, 
                      audio_file: pathlib.Path,
                      source_params: dict,
                      first_sample_time_stamp,
                      aac_bitrate: int = 160000,
                      temp_dir: str = "/tmp") -> zarr.Array

# Single segment extraction
def extract_audio_segment_aac(zarr_group: zarr.Group, 
                             audio_blob_array: zarr.Array,
                             start_sample: int, 
                             end_sample: int, 
                             dtype=np.int16) -> np.ndarray

# Parallel segment extraction  
def parallel_extract_audio_segments_aac(zarr_group: zarr.Group,
                                        audio_blob_array: zarr.Array,
                                        segments: List[Tuple[int, int]],
                                        dtype=np.int16,
                                        max_workers: int = 4) -> List[np.ndarray]
```

#### **PyAV Integration Strategy**
```python
# AAC encoding with PyAV (no subprocess)
import av

def encode_to_aac_pyav(input_file: pathlib.Path, 
                      output_file: pathlib.Path,
                      bitrate: int = 160000,
                      sample_rate: int = 48000) -> dict:
    """Native AAC encoding using PyAV"""
    
def decode_aac_segment_pyav(aac_data: bytes,
                           start_frame: int,
                           frame_count: int) -> np.ndarray:
    """Native AAC decoding using PyAV"""
```

### **Phase 2: AAC Index Module (`aac_index_backend.py`)**

#### **Index Structur**

Some Notes for fixing this structure:

- **Frame flags** don't need to be stored - Each frame is like a key frame. But there is an overlapp: We need some samples from the frame befor to can decode exactly at the beginning of the interesting frame. If we start random access decoding, so we have to start a frame before the start frame to reconstruct the samples exactly. But if we do it, we don't need any frame flags to memorize in the index.
- **Sample count** of the frame – A frame can have 1024 or 960 samples by definition. But ffmpeg can only encode 1024 samples. And in the most cases othe tools can not handle 960 samples. Since we import our audio sources with ffmpeg into the database, we can get only 1024 sample frames! So we don't need this parameter inside the index. But, we can write a 'getter' what gives back always 1024 as requested 'sample count' parameter:
```
def get_aac_frame_samples():
    return 1024  # FFmpeg erzeugt immer 1024-Sample-Frames
```
- **Time stamps** of frame don't need to be stored - If we store the sample position of the first frame, we can calculate the right time stamp exactly by using the sampling frequency:
```
def calculate_timestamp_ms(sample_pos, sample_rate):
    return int(sample_pos * 1000 / sample_rate)
``` 

**Index Structure (3 columns per frame):**
This is the structure as structured Numpy array
```
[byte_offset, frame_size, sample_pos]
     ↓            ↓           ↓      
  Position    Size bytes   Cumulative
  in stream   of frame     sample pos
 (np.uint64) (np.uint16)  (np.uint64)
```
Saved in Zarr Arrays, the Numpy structure is converted to a byte stream with a strong 18 bytes index offset. From Zarr Array back into memory we can address structure elements by start address and offset and we can remap the full byte array into a structured Numpy array.



#### **Index Creation System**
```python
# Main index builder
def build_aac_index(zarr_group: zarr.Group, 
                   audio_blob_array: zarr.Array,
                   use_parallel: bool = True) -> zarr.Array

# Frame analysis functions
def analyze_aac_frames_pyav(aac_data: bytes) -> List[FrameInfo]
def create_frame_index_array(frame_infos: List[FrameInfo]) -> np.ndarray
def optimize_index_for_access(index_array: np.ndarray) -> np.ndarray
```

#### **Frame Analysis Architecture**
```python
@dataclass
class AACFrameInfo:
    """Information about a single AAC frame"""
    frame_index: int          # Sequential frame number
    byte_offset: int          # Position in AAC stream
    frame_size: int           # Size in bytes
    sample_count: int         # Samples in frame (typically 1024)
    sample_position: int      # Cumulative sample position
    timestamp_ms: int         # Time position in milliseconds
    is_keyframe: bool         # True for IDR frames
    bitrate_estimate: int     # Estimated bitrate for this frame
```

#### **Index Storage Format**
```python
# Zarr array structure: (n_frames, 6) uint64 array
INDEX_COLUMNS = 6
COL_BYTE_OFFSET = 0      # uint64: Position in stream
COL_FRAME_SIZE = 1       # uint64: Frame size in bytes  
COL_SAMPLE_COUNT = 2     # uint64: Samples per frame
COL_SAMPLE_POSITION = 3  # uint64: Cumulative sample position
COL_TIMESTAMP_MS = 4     # uint64: Time in milliseconds
COL_FRAME_FLAGS = 5      # uint64: Frame type flags
```

### **Phase 3: Performance Optimization**

#### **Memory Efficiency**
- **Streaming Processing:** Process AAC data in chunks to avoid loading entire files
- **Index Compression:** Use appropriate Zarr chunking for index arrays
- **Native Libraries:** PyAV eliminates subprocess memory overhead
- **Lazy Loading:** Load AAC frames on-demand during extraction

#### **Parallel Processing Strategy**
```python
# Frame analysis can be parallelized
def parallel_frame_analysis(aac_data: bytes, 
                           chunk_size: int = 1000,
                           max_workers: int = 4) -> List[FrameInfo]

# Index creation optimization
def optimize_index_chunks(frame_count: int) -> Tuple[int, int]:
    """Calculate optimal Zarr chunking for index array"""
```

## 📋 **Configuration Parameters**

### **AAC-Specific Configuration**
```python
# Config.py integration points:
class AACConfig:
    # Encoding parameters
    default_bitrate: int = 160000          # Target bitrate
    default_profile: str = "aac_low"       # AAC-LC profile
    variable_bitrate: bool = False         # Use CBR by default
    
    # Index parameters  
    index_chunk_size: int = 5000           # Frames per index chunk
    enable_parallel_indexing: bool = True # Use parallel processing
    max_index_workers: int = 4             # Parallel worker limit
    
    # PyAV parameters
    encoder_preset: str = "medium"         # Encoding speed/quality
    decoder_buffer_size: int = 8192        # Decode buffer size
    enable_native_seeking: bool = True     # Use PyAV seeking
```

### **Performance Tuning**
```python
# Memory management
max_memory_usage_mb: int = 500            # Memory limit for processing
streaming_chunk_size_mb: int = 10         # Chunk size for large files
enable_memory_monitoring: bool = True     # Monitor memory usage

# Quality/Speed tradeoffs
fast_seeking_mode: bool = True            # Optimize for seek speed
quality_priority: str = "balanced"       # balanced/speed/quality
enable_frame_validation: bool = True     # Validate frame integrity
```

## 🚀 **Implementation Roadmap**

### **Phase 1: Core AAC Access (Week 1)**
1. **Create `aac_access.py`** with basic import/export functions
2. **Implement PyAV integration** for encoding/decoding
3. **Test basic AAC import pipeline** with sample files
4. **Validate audio quality** at target bitrates

### **Phase 2: Frame Index System (Week 2)**  
1. **Create `aac_index_backend.py`** with frame analysis
2. **Implement frame detection** using PyAV stream parsing
3. **Build index creation pipeline** with metadata extraction
4. **Test index-based random access** with validation

### **Phase 3: Performance Optimization (Week 3)**
1. **Implement parallel frame analysis** for large files
2. **Optimize memory usage** for streaming processing  
3. **Add comprehensive benchmarking** vs other formats
4. **Performance tuning** based on benchmark results

### **Phase 4: Integration & Testing (Week 4)**
1. **Integration with `aimport.py`** and main API
2. **Comprehensive testing** with various audio types
3. **Performance validation** against targets
4. **Documentation and examples**

## 📊 **Success Metrics**

### **Performance Targets**
- ✅ **Storage Overhead:** <3% additional space vs raw AAC
- ✅ **Random Access Speed:** <25ms for any segment extraction
- ✅ **Import Speed:** Faster than real-time encoding (>1x speed)
- ✅ **Memory Usage:** <500MB peak for any file size
- ✅ **Audio Quality:** Transparent at 160kbps AAC-LC

### **Compatibility Goals**
- ✅ **Input Formats:** Support all formats supported by PyAV
- ✅ **Sample Rates:** 44.1kHz, 48kHz, 96kHz support
- ✅ **Channel Counts:** Mono, Stereo, 5.1 support
- ✅ **Bitrates:** 128kbps - 320kbps range
- ✅ **Platform Support:** Linux, macOS, Windows compatibility

### **Quality Benchmarks**
- ✅ **vs MP3 320kbps:** Better quality at 160kbps AAC
- ✅ **vs Opus 160kbps:** Comparable quality, much better random access
- ✅ **vs FLAC:** 32% smaller, similar access speed, lossy
- ✅ **Perceptual Quality:** No audible artifacts in target use cases

## 🔧 **Risk Mitigation**

### **Technical Risks**
- **PyAV Version Compatibility:** Test with multiple PyAV versions
- **AAC Frame Boundary Detection:** Validate frame parsing accuracy
- **Large File Handling:** Test with multi-GB files
- **Memory Leaks:** Monitor long-running operations

### **Performance Risks**  
- **Index Size Growth:** Monitor overhead with various audio lengths
- **Seek Accuracy:** Validate sample-accurate positioning
- **Parallel Processing Scaling:** Test with different CPU counts
- **I/O Bottlenecks:** Optimize Zarr access patterns

### **Fallback Strategies**
- **Sequential Fallback:** If parallel processing fails
- **Memory Constraints:** Streaming mode for large files
- **Quality Degradation:** Lower bitrate options for storage constraints
- **Compatibility Issues:** Alternative encoding parameters

## 📋 **Dependencies & Requirements**

### **Core Dependencies**
```python
# Required packages
av >= 10.0.0                    # PyAV for native AAC support
zarr >= 2.10.0                  # Storage backend
numpy >= 1.20.0                 # Array operations
soundfile >= 0.10.0             # Audio I/O fallback
concurrent.futures              # Parallel processing (stdlib)
```

### **Optional Enhancements**
```python
# Performance monitoring
psutil >= 5.8.0                 # Memory monitoring
tqdm >= 4.60.0                  # Progress bars
multiprocessing                 # Parallel processing (stdlib)

# Development/Testing
pytest >= 6.0.0                # Testing framework
librosa >= 0.8.0               # Audio analysis validation
matplotlib >= 3.3.0           # Performance visualization
```

## 🎯 **Expected Outcomes**

### **Primary Benefits**
1. **Storage Efficiency:** 57% smaller than original, 32% smaller than FLAC
2. **Access Performance:** 40-65x faster than sequential processing
3. **Implementation Simplicity:** No complex state management required
4. **Universal Compatibility:** AAC works everywhere
5. **Future-Proof:** Mature, standardized technology

### **Comparison with Alternatives**
```
Format Comparison Summary:
┌─────────────────┬─────────────┬─────────────┬──────────────┬─────────────┐
│ Format          │ Size (7min) │ Overhead    │ Access Speed │ Complexity  │
├─────────────────┼─────────────┼─────────────┼──────────────┼─────────────┤
│ AAC + Index     │ 8.8 MB      │ 2.3%        │ ~20ms       │ Low         │
│ Opus + Index    │ 12.8 MB     │ 49%         │ ~1ms        │ Very High   │
│ MP3 + Index     │ 8.7 MB      │ 1%          │ ~25ms       │ Low         │
│ FLAC + Index    │ 13 MB       │ minimal     │ ~15ms       │ Medium      │
└─────────────────┴─────────────┴─────────────┴──────────────┴─────────────┘
```

**AAC-LC emerges as the optimal balance of storage efficiency, access speed, and implementation simplicity.**

## **Boundary Parameters**

- The main API for users is ```aimport.py``` for importing only (not for read access of audio or attributes)
- The AAC implementation should be similar as demonstrated by 'flac' import with ```aimport.py```, ```flac_access.py``` and ```flac_indes_backend.py```.
- Programm configurations are done in ```Config``` class of ```config.py```. We have to add new configuration values which are related to AAC with an ```aac_``` prefix. **Never remove a non-```aac_``` parameter**
- With ```Config.set()```parameters can be changed during the application runs. So different test scenarios can be implemented by changing such parameters.
- By using ```logsetup.py``` specialized loguru messaging is possible and can be used for debugging. In particular ```'DEBUG'``` and ```'TRACE'``` level are usable for debugging. All other levels are for production use and should only contain corresponding messages. ```'SUCCESS'``` level should be used only by finishing procedures which are application near.
- During implementation, the main target files are ```aac_access.py``` and ```aac_index_backend.py```. But also the ```Config``` class (```config.py```) can be extended and ```aimport.py``` should be extended. If there are 'Opus' related content in ```aimport.py``` so this should removed / replaced by AAC related functions completely.
- **Don't never change the common and flac-depending content of ```aimport.py```**.
- In all module files (```aac_...py```) include the logger start procedure:

```
# import and initialize logging
from .logsetup import get_module_logger
logger = get_module_logger(__file__)
logger.trace("<WRITE WHAT WE LOAD> loading...")
```

and replace ```<WRITE WHAT WE LOAD> ``` with senfull content.

## **Testing conditions**

- **Test directory** for the test script is ./tests/
- **Audio Source Files** for test runs can be found in project folder ./tests/testdate/
- **Target Directory** for Zarr data base (version 3) and test result files is ./tests/testresults/
- **Cleaning Up** at each test start is mandatory: First remove all content in ./tests/testresults/
- **Zarr Database Structur** is fixed for the import range of audio data. Following is an example to set up the structur
- **PyTest** - All official tests must have a structure to be a pytest compatible test. So we can use the same tests also for CI. Tests may have additional prints for debugging outputs which should be communicated with [Claude.ai](claude.ai). But: Special debug tests from claude.ai what we use for finding current errors don't need this structure. Such tests has to be short and straight forward.  

### Helpers for all tests

Following some common function to can be used in all test files:

```
import unittest
import pathlib
from typing import List
import zarr
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
```

---

## Anweisung für Claude.ai

- Für Änderungen in den bestehenden Dateien, schreibe entweder komplette Datei-Artifacts mit den Änderungen oder einzelne Funktionen mit einer Zusatzerklärung, was ausgetauscht werden muss. Bitte schreibe keine Scripte, die die Quellfiles "in-line" manipulieren. Entweder Du gibst das gesamte File als Artifact zurück oder übergibst mir genau beschriebene Bereiche, die ich dann selbst austausche.
- Vermeide Icons, Emojis oder sonstiges in Programmen und loggings. In Konzept- und Statusfiles sind sie erlaubt. In Testfiles sind grundlegende Symbole für Erfolg und Fehler oder Warnung und ähnliches erlaubt, soweit das die schnelle Aufnahme der Information durch den Leser fördert.
- Die Kommunikationssprache ist deutsch. In Software wird aber ausschließlich in English kommentiert und Ausgaben, Logs, Fehlermeldungen erzeugt.




**This concept provides the foundation for a production-ready AAC random access system that surpasses the Opus approach in storage efficiency while maintaining excellent performance.**