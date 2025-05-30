#!/usr/bin/env python3
"""
Opus I/O Debug Tool
===================

CRITICAL: Diagnose the massive I/O problem during Opus indexing
Problem: Several 100 MB read+write I/O per second during indexing

This tool will:
1. Monitor I/O during Opus indexing with detailed statistics
2. Identify which components cause excessive I/O
3. Test different parallelization strategies
4. Provide recommendations for fixing the I/O bottleneck
"""

import time
import psutil
import threading
import pathlib
import datetime
import sys
import os
import numpy as np

# Import the problematic modules
try:
    import zarrwlr
    from zarrwlr import opus_index_backend
    import zarr
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class IOMonitor:
    """Real-time I/O monitoring during Opus operations"""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.io_data = []
        self.monitor_thread = None
        self.start_time = None
        
    def start_monitoring(self):
        """Start I/O monitoring in background thread"""
        self.monitoring = True
        self.start_time = time.time()
        self.io_data = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"🔍 I/O monitoring started (interval: {self.interval}s)")
    
    def stop_monitoring(self):
        """Stop I/O monitoring and return statistics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        if not self.io_data:
            return {}
        
        # Calculate statistics
        total_time = time.time() - self.start_time
        read_bytes = [d['read_bytes'] for d in self.io_data]
        write_bytes = [d['write_bytes'] for d in self.io_data]
        
        # Calculate deltas (actual I/O per interval)
        read_deltas = [read_bytes[i] - read_bytes[i-1] for i in range(1, len(read_bytes))]
        write_deltas = [write_bytes[i] - write_bytes[i-1] for i in range(1, len(write_bytes))]
        
        if read_deltas and write_deltas:
            stats = {
                'total_time': total_time,
                'samples': len(self.io_data),
                'total_read_mb': (read_bytes[-1] - read_bytes[0]) / 1024 / 1024,
                'total_write_mb': (write_bytes[-1] - write_bytes[0]) / 1024 / 1024,
                'avg_read_mb_per_sec': np.mean(read_deltas) / self.interval / 1024 / 1024,
                'avg_write_mb_per_sec': np.mean(write_deltas) / self.interval / 1024 / 1024,
                'peak_read_mb_per_sec': max(read_deltas) / self.interval / 1024 / 1024,
                'peak_write_mb_per_sec': max(write_deltas) / self.interval / 1024 / 1024,
                'raw_data': self.io_data
            }
        else:
            stats = {'error': 'No valid I/O deltas calculated'}
        
        print(f"📊 I/O monitoring stopped after {total_time:.2f}s")
        return stats
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                io_counters = process.io_counters()
                timestamp = time.time() - self.start_time
                
                self.io_data.append({
                    'timestamp': timestamp,
                    'read_bytes': io_counters.read_bytes,
                    'write_bytes': io_counters.write_bytes,
                    'read_count': io_counters.read_count,
                    'write_count': io_counters.write_count
                })
                
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")
                break

class OpusIOTester:
    """Test different Opus indexing strategies and measure I/O"""
    
    def __init__(self):
        self.test_results_dir = pathlib.Path(__file__).parent.resolve() / "testresults"
        self.test_results_dir.mkdir(exist_ok=True)
        
    def find_test_file(self) -> pathlib.Path:
        """Find test audio file for I/O testing"""
        test_files = [
            "testdata/audiomoth_long_snippet_converted.opus",
            "testdata/bird1_snippet.mp3",
            "testdata/audiomoth_short_snippet.wav",
        ]
        
        base_path = pathlib.Path(__file__).parent.resolve()
        for test_file in test_files:
            test_path = base_path / test_file
            if test_path.exists():
                file_size_mb = test_path.stat().st_size / 1024 / 1024
                print(f"📁 Using test file: {test_path.name} ({file_size_mb:.2f} MB)")
                return test_path
        
        raise FileNotFoundError("No test files found")
    
    def test_sequential_vs_parallel_io(self):
        """Compare I/O behavior between sequential and parallel indexing"""
        print("=" * 80)
        print("OPUS I/O ANALYSIS - Sequential vs Parallel")
        print("=" * 80)
        
        audio_file = self.find_test_file()
        
        # Test 1: Sequential indexing I/O
        print("\n🔍 TEST 1: Sequential Indexing I/O")
        sequential_stats = self._test_indexing_io(audio_file, use_parallel=False)
        
        # Test 2: Parallel indexing I/O  
        print("\n🔍 TEST 2: Parallel Indexing I/O")
        parallel_stats = self._test_indexing_io(audio_file, use_parallel=True)
        
        # Comparison
        print("\n" + "=" * 60)
        print("I/O COMPARISON RESULTS")
        print("=" * 60)
        
        self._print_io_comparison(sequential_stats, parallel_stats)
        
        # Recommendations
        self._provide_io_recommendations(sequential_stats, parallel_stats)
    
    def _test_indexing_io(self, audio_file: pathlib.Path, use_parallel: bool) -> dict:
        """Test I/O during indexing with specified mode"""
        mode = "PARALLEL" if use_parallel else "SEQUENTIAL"
        print(f"Testing {mode} indexing I/O...")
        
        # Create temporary Zarr store
        timestamp = int(time.time())
        test_store_dir = self.test_results_dir / f"zarr3-store-io-test-{mode.lower()}-{timestamp}"
        
        try:
            # Create audio group and import
            audio_group = zarrwlr.create_original_audio_group(
                store_path=test_store_dir,
                group_path='audio_imports'
            )
            
            print(f"  📥 Importing audio file...")
            import_monitor = IOMonitor(interval=0.05)  # High frequency monitoring
            import_monitor.start_monitoring()
            
            # Import with specified mode
            zarrwlr.import_original_audio_file(
                audio_file=audio_file,
                zarr_original_audio_group=audio_group,
                first_sample_time_stamp=datetime.datetime.now(),
                target_codec='opus',
                opus_bitrate=128000,
                temp_dir=self.test_results_dir
            )
            
            import_stats = import_monitor.stop_monitoring()
            
            # Get the imported group
            group_names = [name for name in audio_group.keys() if name.isdigit()]
            if not group_names:
                raise RuntimeError("No audio groups found after import")
            
            latest_group_name = max(group_names, key=int)
            imported_group = audio_group[latest_group_name]
            
            # Check if parallel was actually used
            opus_index = imported_group['opus_index']
            actual_parallel = opus_index.attrs.get('parallel_processing_used', False)
            
            import_stats['requested_parallel'] = use_parallel
            import_stats['actual_parallel'] = actual_parallel
            import_stats['index_pages'] = opus_index.shape[0]
            import_stats['audio_size_mb'] = imported_group['audio_data_blob_array'].shape[0] / 1024 / 1024
            
            print(f"  ✅ Import completed:")
            print(f"     Requested: {mode}")
            print(f"     Actual: {'PARALLEL' if actual_parallel else 'SEQUENTIAL'}")
            print(f"     Pages: {import_stats['index_pages']}")
            print(f"     Audio: {import_stats['audio_size_mb']:.1f} MB")
            
            return import_stats
            
        except Exception as e:
            print(f"  ❌ Import failed: {e}")
            return {'error': str(e)}
        
        finally:
            # Clean up
            if test_store_dir.exists():
                import shutil
                shutil.rmtree(test_store_dir)
    
    def _print_io_comparison(self, sequential_stats: dict, parallel_stats: dict):
        """Print detailed I/O comparison"""
        
        def print_stats(name: str, stats: dict):
            if 'error' in stats:
                print(f"{name}: ❌ {stats['error']}")
                return
            
            print(f"{name}:")
            print(f"  Total Time: {stats.get('total_time', 0):.2f}s")
            print(f"  Total Read: {stats.get('total_read_mb', 0):.1f} MB")
            print(f"  Total Write: {stats.get('total_write_mb', 0):.1f} MB")
            print(f"  Avg Read Rate: {stats.get('avg_read_mb_per_sec', 0):.1f} MB/s")
            print(f"  Avg Write Rate: {stats.get('avg_write_mb_per_sec', 0):.1f} MB/s")
            print(f"  Peak Read Rate: {stats.get('peak_read_mb_per_sec', 0):.1f} MB/s")
            print(f"  Peak Write Rate: {stats.get('peak_write_mb_per_sec', 0):.1f} MB/s")
            print(f"  Index Pages: {stats.get('index_pages', 0)}")
            print(f"  Audio Size: {stats.get('audio_size_mb', 0):.1f} MB")
            print(f"  Parallel Used: {stats.get('actual_parallel', False)}")
        
        print_stats("📈 SEQUENTIAL", sequential_stats)
        print()
        print_stats("🚀 PARALLEL", parallel_stats)
        
        # Calculate differences
        if 'error' not in sequential_stats and 'error' not in parallel_stats:
            print(f"\n📊 DIFFERENCES:")
            time_ratio = parallel_stats.get('total_time', 1) / sequential_stats.get('total_time', 1)
            read_ratio = parallel_stats.get('total_read_mb', 0) / max(sequential_stats.get('total_read_mb', 1), 1)
            write_ratio = parallel_stats.get('total_write_mb', 0) / max(sequential_stats.get('total_write_mb', 1), 1)
            
            print(f"  Time Ratio (P/S): {time_ratio:.2f}x")
            print(f"  Read Ratio (P/S): {read_ratio:.2f}x")
            print(f"  Write Ratio (P/S): {write_ratio:.2f}x")
    
    def _provide_io_recommendations(self, sequential_stats: dict, parallel_stats: dict):
        """Provide recommendations based on I/O analysis"""
        print(f"\n🔧 RECOMMENDATIONS:")
        
        if 'error' in parallel_stats:
            print("❌ Parallel processing failed - focus on fixing parallel implementation")
            return
        
        # Check for excessive I/O
        parallel_peak_read = parallel_stats.get('peak_read_mb_per_sec', 0)
        parallel_peak_write = parallel_stats.get('peak_write_mb_per_sec', 0)
        
        if parallel_peak_read > 100 or parallel_peak_write > 100:
            print(f"🚨 CRITICAL: Excessive I/O detected!")
            print(f"   Peak Read: {parallel_peak_read:.1f} MB/s")
            print(f"   Peak Write: {parallel_peak_write:.1f} MB/s")
            print(f"   Recommended fixes:")
            print(f"   1. Reduce chunk overlap (currently 1KB)")
            print(f"   2. Use ThreadPoolExecutor instead of ProcessPoolExecutor")
            print(f"   3. Implement single Zarr-store access pattern")
            print(f"   4. Add chunking size optimization")
        
        # Check for parallel efficiency
        if parallel_stats.get('actual_parallel', False):
            seq_time = sequential_stats.get('total_time', 1)
            par_time = parallel_stats.get('total_time', 1)
            speedup = seq_time / par_time
            
            if speedup < 1.2:
                print(f"⚠️  Low parallel efficiency: {speedup:.2f}x speedup")
                print(f"   Possible causes:")
                print(f"   1. I/O bottleneck dominates CPU parallelization")
                print(f"   2. Zarr-store contention between workers")
                print(f"   3. Excessive memory copying in workers")
        
        print(f"\n💡 SUGGESTED IMMEDIATE FIXES:")
        print(f"   1. Replace ProcessPoolExecutor with ThreadPoolExecutor")
        print(f"   2. Reduce chunk_size_mb from 4MB to 1MB")
        print(f"   3. Remove chunk overlap (set overlap=0)")
        print(f"   4. Implement single shared Zarr access")

def main():
    print("🔍 OPUS I/O BOTTLENECK ANALYZER")
    print("=" * 50)
    print("Investigating excessive I/O during Opus indexing...")
    
    if not psutil:
        print("❌ psutil not available - cannot monitor I/O")
        sys.exit(1)
    
    try:
        tester = OpusIOTester()
        tester.test_sequential_vs_parallel_io()
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()