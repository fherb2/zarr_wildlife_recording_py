# ZARRWLR Project Context Documentation

## Project Overview

**zarrwlr** (Zarr Wildlife Recording) is a sophisticated Python package for importing, storing, and accessing audio data in Zarr v3 format, specifically optimized for wildlife and scientific audio recording applications.

### Core Architecture

The project implements a **three-layer architecture**:

1. **High-Level API** (`aimport.py`) - User-friendly import interface with intelligent automation
2. **Analysis Engine** (`import_utils.py`) - Intelligent audio file analysis and parameter suggestion system  
3. **Storage Backends** (`flac_access.py`, `aac_access.py`) - Codec-specific optimized storage implementations

### Key Design Principles

- **Intelligence First**: Automatic quality analysis and optimal format suggestions
- **Wildlife-Optimized**: Special handling for ultrasound recordings (bat calls, >96kHz)
- **Scientific Accuracy**: Preserves spatial information for multichannel recordings
- **Performance-Focused**: Copy-mode detection, parallel processing, memory-efficient streaming
- **Type Safety**: Comprehensive validation and conflict detection

## Technical Specializations

### Audio Format Intelligence

**Source Quality Classification:**
- **Uncompressed**: PCM variants (WAV, AIFF)
- **Lossless**: FLAC, ALAC with perfect preservation
- **Lossy**: MP3, AAC with intelligent upgrade strategies

**Target Format Optimization:**
- **FLAC**: 1Hz-655kHz range, lossless compression levels 0-12
- **AAC-LC**: 32-320kbps, optimized for wildlife recording standards
- **Copy-Mode**: 1:1 transfer without re-encoding when possible

### Specialized Scenarios

**Ultrasound Processing (>96kHz):**
- Automatic detection and protection
- **Reinterpretation** (not resampling) preserves frequency domain characteristics
- Essential for bat echolocation research

**Multichannel Scientific Audio:**
- FLAC: Perfect spatial preservation up to 8 channels
- AAC: Warnings about spatial information loss for >2 channels
- Microphone array support with conflict detection

**Quality-Aware Conversions:**
- Adaptive bitrate calculation (+25-40% upgrades based on source quality)
- Minimum scientific standards (160kbps mono, 190kbps stereo)
- Three-tier conflict detection (blocking/quality/efficiency)

## Import Workflow Architecture

### 1. File Analysis (`FileParameter` class)
- **Deep Technical Analysis**: ffprobe-based stream analysis with metadata extraction
- **Quality Assessment**: Source compression type, bitrate classification, quality tier determination
- **Intelligent Suggestions**: Format-specific recommendations based on audio characteristics
- **Conflict Detection**: Multi-level validation preventing data loss scenarios
- **Beautiful Terminal Output**: User-friendly analysis display with API access hints

### 2. Status Checking (`is_audio_in_zarr_audio_group`)
- **Duplicate Detection**: SHA256-based hash comparison
- **Batch Processing**: Parallel analysis with configurable worker counts
- **Performance Optimization**: Analysis-only operations, no full file loading

### 3. Import Execution (`aimport`)
- **Intelligent Import**: Uses FileParameter analysis and configuration
- **Parallel Processing**: File-level parallelization for large operations
- **Error Handling**: Comprehensive rollback on failure with detailed error reporting
- **Performance Tracking**: Detailed timing and compression metrics

## Test Architecture

### Test Structure
```
./tests/
├── testdata/           # Input: Real audio files for testing
├── testresults/        # Output: All test artifacts (cleaned each run)
├── test_import_utils.py   # Analysis engine tests
├── test_aimport.py        # High-level API tests  
├── run_all_tests.py       # Combined test runner
└── test_utils.py          # Test management utilities
```

### Test Categories

**Core Functionality Tests:**
- Enum conversions and codec classification
- Copy-mode detection and benefits analysis
- Quality analyzer intelligence testing
- Conflict detection validation

**Real File Integration Tests:**
- All available testdata categories (uncompressed/lossless/lossy/multichannel/ultrasound)
- End-to-end workflows (analysis → import → verification)
- Performance benchmarks and memory efficiency

**Specialized Scenario Tests:**
- Ultrasound protection (resampling blocking for >96kHz)
- Multichannel scientific recording workflows
- Copy-mode vs re-encoding decision trees
- Large file handling and batch operations

### Test Isolation Strategy

**Unique Test Environments:**
- Each test gets microsecond-timestamped directory in `testresults/`
- Complete Zarr store isolation prevents cross-test interference
- Persistent artifacts enable post-failure debugging

**Performance Targets:**
- FileParameter analysis: <5s for normal files
- Small file imports (<1MB): <30s
- Large file imports (200MB): <120s
- Batch operations: <60s average per file

## Configuration System

**Runtime-Configurable Parameters:**
- Global and module-specific log levels with automatic reconfiguration
- AAC processing settings (bitrates, quality presets, worker counts)
- Zarr storage optimization (chunk sizes, compression levels)
- Import batch processing parameters

**YAML Serialization:**
- Export/import with comment preservation
- Type-safe validation with intelligent error messages
- Development template for easy extension

## Development Context

### Code Quality Standards
- **Type Safety**: Comprehensive type hints and validation
- **Error Handling**: Graceful degradation with detailed error messages
- **Documentation**: NumPy-style docstrings with usage examples
- **Performance**: Memory-efficient streaming with progress tracking

### Wildlife Research Optimizations
- **Bat Echolocation**: Specialized ultrasound reinterpretation techniques
- **Long-Duration Monitoring**: Efficient chunking for continuous recordings
- **Multichannel Arrays**: Spatial preservation for directional analysis
- **Field Robustness**: Comprehensive error recovery and validation

### Integration Points
- **ffmpeg/ffprobe**: External tool integration with validation
- **PyAV**: High-performance AAC processing with ADTS format optimization
- **Zarr v3**: Next-generation chunked storage with universal serialization
- **Loguru**: Advanced logging with performance optimization

## Future Development Areas

**Planned Enhancements:**
- NUMPY format implementation for unlimited sample rate support
- Enhanced parallel processing with GPU acceleration hooks
- Real-time streaming import capabilities
- Advanced metadata preservation and scientific provenance tracking

**Extension Points:**
- New codec support through backend architecture
- Custom analysis algorithms via plugin system
- Cloud storage backends (S3, GCS) integration
- Workflow automation for field station deployments

This context enables continued development while maintaining the sophisticated balance of automation, performance, and scientific accuracy that characterizes the zarrwlr project.
