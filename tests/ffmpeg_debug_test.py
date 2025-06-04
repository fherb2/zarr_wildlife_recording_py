#!/usr/bin/env python3
"""
Standalone ffmpeg Debug Test
===========================

Test ffmpeg command separately to find the hanging issue
"""

import pathlib
import subprocess
import tempfile
import time
import os

def test_ffmpeg_standalone():
    """Test the exact ffmpeg command that's hanging"""
    
    print("🔍 FFMPEG STANDALONE DEBUG TEST")
    print("=" * 50)
    
    # Find test file
    test_data_dir = pathlib.Path(__file__).parent / "testdata"
    test_file = test_data_dir / "audiomoth_long_snippet.wav"
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    print(f"📁 Input file: {test_file}")
    print(f"📏 Input file size: {test_file.stat().st_size} bytes")
    
    # Check file permissions
    if not os.access(test_file, os.R_OK):
        print("❌ Input file not readable!")
        return False
    else:
        print("✅ Input file is readable")
    
    # Create temp output file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.opus') as tmp_out:
        output_file = pathlib.Path(tmp_out.name)
    
    print(f"📝 Output file: {output_file}")
    
    try:
        # Test 1: ffmpeg version (should be fast)
        print("\n🧪 TEST 1: ffmpeg version")
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, timeout=5, text=True)
            print(f"✅ ffmpeg version check: {result.returncode}")
            print(f"   Version: {result.stdout.split()[2] if result.stdout else 'unknown'}")
        except Exception as e:
            print(f"❌ ffmpeg version failed: {e}")
            return False
        
        # Test 2: Input file analysis (should be fast)
        print("\n🧪 TEST 2: Input file analysis")
        try:
            ffprobe_cmd = ['ffprobe', '-hide_banner', '-loglevel', 'error', 
                          '-show_format', '-show_streams', str(test_file)]
            result = subprocess.run(ffprobe_cmd, capture_output=True, timeout=10, text=True)
            print(f"✅ ffprobe analysis: {result.returncode}")
            if result.stdout:
                print(f"   Format info: {result.stdout[:200]}")
        except Exception as e:
            print(f"❌ ffprobe failed: {e}")
            print("⚠️ Input file might be corrupted")
        
        # Test 3: Simple ffmpeg command (no Opus, just copy)
        print("\n🧪 TEST 3: Simple copy test")
        simple_output = output_file.with_suffix('.wav')
        try:
            simple_cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                         '-i', str(test_file), '-t', '1', str(simple_output)]
            print(f"   Command: {' '.join(simple_cmd)}")
            
            start_time = time.time()
            result = subprocess.run(simple_cmd, capture_output=True, timeout=10, text=True)
            duration = time.time() - start_time
            
            print(f"✅ Simple copy: {result.returncode} in {duration:.3f}s")
            if simple_output.exists():
                print(f"   Output size: {simple_output.stat().st_size} bytes")
                simple_output.unlink()  # cleanup
                
        except subprocess.TimeoutExpired:
            print("⏰ Simple copy TIMED OUT - ffmpeg has serious issues!")
            return False
        except Exception as e:
            print(f"❌ Simple copy failed: {e}")
        
        # Test 4: Check libopus availability
        print("\n🧪 TEST 4: Opus codec availability")
        try:
            codec_cmd = ['ffmpeg', '-hide_banner', '-encoders']
            result = subprocess.run(codec_cmd, capture_output=True, timeout=5, text=True)
            if 'libopus' in result.stdout:
                print("✅ libopus encoder available")
            else:
                print("❌ libopus encoder NOT available!")
                print("   Available encoders with 'opus':")
                for line in result.stdout.split('\n'):
                    if 'opus' in line.lower():
                        print(f"   {line}")
                return False
        except Exception as e:
            print(f"❌ Codec check failed: {e}")
        
        # Test 5: The actual hanging command (with shorter duration)
        print("\n🧪 TEST 5: Actual Opus conversion (short)")
        try:
            opus_cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'info',  # Change to 'info' for more output
                       '-i', str(test_file), '-t', '2',  # Only 2 seconds
                       '-c:a', 'libopus', '-b:a', '128000', 
                       '-f', 'opus', str(output_file)]
            
            print(f"   Command: {' '.join(opus_cmd)}")
            print("   Starting conversion...")
            
            start_time = time.time()
            
            # Use Popen for real-time output monitoring
            process = subprocess.Popen(opus_cmd, 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT,  # Combine stderr with stdout
                                     text=True, universal_newlines=True)
            
            # Monitor output in real-time
            output_lines = []
            while True:
                line = process.stdout.readline()
                if line:
                    output_lines.append(line.strip())
                    print(f"   ffmpeg: {line.strip()}")
                
                # Check if process finished
                if process.poll() is not None:
                    break
                
                # Timeout check
                if time.time() - start_time > 15:
                    print("⏰ Timeout reached, killing process...")
                    process.terminate()
                    process.wait(timeout=2)
                    return False
            
            duration = time.time() - start_time
            return_code = process.returncode
            
            print(f"✅ Opus conversion: {return_code} in {duration:.3f}s")
            
            if output_file.exists():
                size = output_file.stat().st_size
                print(f"   Output size: {size} bytes")
                
                if size > 0:
                    print("✅ ffmpeg successfully created Opus output!")
                    
                    # Quick validation
                    with open(output_file, 'rb') as f:
                        first_bytes = f.read(100)
                        if b'OpusHead' in first_bytes:
                            print("✅ Valid Opus file created (OpusHead found)")
                        else:
                            print("⚠️ Opus file created but no OpusHead found")
                            print(f"   First 50 bytes: {first_bytes[:50]}")
                else:
                    print("❌ ffmpeg created empty file")
                    return False
            else:
                print("❌ ffmpeg did not create output file")
                return False
            
        except Exception as e:
            print(f"❌ Opus conversion failed: {e}")
            return False
        
        print("\n✅ ALL TESTS PASSED - ffmpeg should work!")
        return True
        
    finally:
        # Cleanup
        if output_file.exists():
            output_file.unlink()

if __name__ == "__main__":
    success = test_ffmpeg_standalone()
    if success:
        print("\n🎉 ffmpeg works correctly!")
        print("The hanging issue must be elsewhere...")
    else:
        print("\n💥 ffmpeg has issues!")
        print("This explains the hanging in import_opus_to_zarr()")
    
    exit(0 if success else 1)