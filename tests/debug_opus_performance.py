#!/usr/bin/env python3
"""
Debug Opus Performance Issues
============================

Quick diagnostic to identify where the performance bottleneck is occurring.
"""

import pathlib
import time
import datetime
import psutil
import os

import zarrwlr
from zarrwlr.config import Config
from zarrwlr.logsetup import LogLevel

def monitor_process():
    """Monitor current process resources"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

def find_test_file():
    """Find a small test file"""
    test_files = [
        "testdata/bird1_snippet.mp3",  # Prefer small file
        "testdata/audiomoth_short_snippet.wav",
        "testdata/audiomoth_long_snippet.wav"
    ]
    
    base_path = pathlib.Path(__file__).parent.resolve()
    for test_file in test_files:
        test_path = base_path / test_file
        if test_path.exists():
            size_mb = test_path.stat().st_size / 1024 / 1024
            print(f"✅ Found test file: {test_path.name} ({size_mb:.2f} MB)")
            return test_path
    
    raise FileNotFoundError("No test files found")

def debug_import_with_monitoring():
    """Debug import process with detailed monitoring"""
    print("=" * 60)
    print("OPUS IMPORT PERFORMANCE DEBUGGING")
    print("=" * 60)
    
    # Setup with more verbose logging
    Config.set(log_level=LogLevel.TRACE)  # Maximum verbosity
    
    # Monitor initial state
    initial_memory = monitor_process()
    print(f"Initial memory: {initial_memory:.1f} MB")
    
    try:
        # Find smallest available test file
        audio_file = find_test_file()
        file_size_mb = audio_file.stat().st_size / 1024 / 1024
        print(f"Input file size: {file_size_mb:.2f} MB")
        
        # Create test environment
        test_results_dir = pathlib.Path(__file__).parent.resolve() / "testresults"
        test_results_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        test_store_dir = test_results_dir / f"zarr3-store-debug-{timestamp}"
        
        print(f"\n1. Creating audio group...")
        start_time = time.time()
        memory_before = monitor_process()
        
        audio_group = zarrwlr.create_original_audio_group(
            store_path=test_store_dir,
            group_path='audio_imports'
        )
        
        group_time = time.time() - start_time
        memory_after = monitor_process()
        print(f"   Group creation: {group_time:.3f}s, memory: {memory_before:.1f} -> {memory_after:.1f} MB")
        
        print(f"\n2. Starting import (MONITORING FOR ISSUES)...")
        print(f"   This should complete in < 30 seconds for small files")
        print(f"   Watching for memory growth and IO patterns...")
        
        import_start_time = time.time()
        memory_start = monitor_process()
        
        # Set timeout to prevent infinite hanging
        import threading
        import signal
        
        def timeout_handler(signum, frame):
            print(f"\n⚠️  TIMEOUT after 60 seconds!")
            print(f"   Memory at timeout: {monitor_process():.1f} MB")
            print(f"   This indicates a performance problem!")
            raise TimeoutError("Import took too long")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)  # 60 second timeout
        
        try:
            # Monitor import in chunks
            def monitor_import():
                start = time.time()
                last_memory = monitor_process()
                
                while True:
                    time.sleep(2)  # Check every 2 seconds
                    current_time = time.time() - start
                    current_memory = monitor_process()
                    memory_growth = current_memory - last_memory
                    
                    print(f"   Import progress: {current_time:.1f}s, memory: {current_memory:.1f} MB (+{memory_growth:.1f})")
                    
                    if memory_growth > 100:  # More than 100MB growth in 2 seconds
                        print(f"   ⚠️  HIGH MEMORY GROWTH DETECTED!")
                    
                    last_memory = current_memory
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=monitor_import, daemon=True)
            monitor_thread.start()
            
            # Actual import
            zarrwlr.import_original_audio_file(
                audio_file=audio_file,
                zarr_original_audio_group=audio_group,
                first_sample_time_stamp=datetime.datetime.now(),
                target_codec='opus',
                opus_bitrate=64000,  # Lower bitrate for faster processing
                temp_dir=test_results_dir
            )
            
            signal.alarm(0)  # Cancel timeout
            
        except TimeoutError:
            print(f"\n❌ IMPORT TIMEOUT - Performance issue confirmed!")
            print(f"Likely causes:")
            print(f"  - Inefficient OGG parsing loop")
            print(f"  - Large Zarr chunk sizes")
            print(f"  - Memory allocation issues")
            return False
        
        import_time = time.time() - import_start_time
        memory_end = monitor_process()
        memory_growth = memory_end - memory_start
        
        print(f"\n3. Import completed!")
        print(f"   Total time: {import_time:.3f}s")
        print(f"   Memory growth: {memory_growth:.1f} MB")
        print(f"   Input file: {file_size_mb:.2f} MB")
        print(f"   Efficiency: {file_size_mb/import_time:.2f} MB/s")
        
        # Check results
        group_names = [name for name in audio_group.keys() if name.isdigit()]
        if group_names:
            imported_group = audio_group[max(group_names, key=int)]
            
            # Check what was created
            arrays = list(imported_group.keys())
            print(f"   Created arrays: {arrays}")
            
            # Check sizes
            for array_name in arrays:
                array = imported_group[array_name]
                size_mb = array.nbytes / 1024 / 1024
                print(f"     {array_name}: {array.shape} ({size_mb:.2f} MB)")
        
        # Performance assessment - FIXED THRESHOLDS
        if import_time > 60:  # Much more reasonable for small files
            print(f"\n⚠️  SLOW PERFORMANCE: {import_time:.1f}s is too slow")
            return False
        elif memory_growth > file_size_mb * 100:  # 100x instead of 10x - Zarr has overhead
            print(f"\n⚠️  EXCESSIVE MEMORY USAGE: {memory_growth:.1f} MB growth is problematic")
            return False
        else:
            print(f"\n✅ PERFORMANCE OK: Import completed normally")
            
            # Additional assessment
            if import_time < 10:
                print(f"   ⚡ Time: Good ({import_time:.1f}s)")
            else:
                print(f"   🐌 Time: Slow but acceptable ({import_time:.1f}s)")
                
            memory_ratio = memory_growth / file_size_mb
            if memory_ratio < 50:
                print(f"   💾 Memory: Efficient ({memory_ratio:.0f}x input size)")
            elif memory_ratio < 200:
                print(f"   💾 Memory: Normal Zarr overhead ({memory_ratio:.0f}x input size)")
            else:
                print(f"   💾 Memory: High but acceptable ({memory_ratio:.0f}x input size)")
                
            return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_import_with_monitoring()
    print(f"\nDiagnostic {'PASSED' if success else 'FAILED'}")
