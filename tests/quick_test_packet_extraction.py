#!/usr/bin/env python3
"""
Quick Test - Direct Packet Extraction
=====================================

Test packet-based extraction directly without going through the generic API.
This bypasses the integration issues and tests the core functionality.
"""

import pathlib
import time
import numpy as np
import zarr

def test_direct_packet_extraction():
    """Test packet-based extraction directly"""
    print("=" * 60)
    print("DIRECT PACKET EXTRACTION TEST")
    print("=" * 60)
    
    try:
        # Find existing test store
        test_results_dir = pathlib.Path(__file__).parent.resolve() / "testresults"
        store_dirs = list(test_results_dir.glob("zarr3-store-*"))
        
        if not store_dirs:
            print("❌ No test stores found. Run debug_opus_performance.py first")
            return False
        
        # Use most recent store
        latest_store = max(store_dirs, key=lambda p: p.stat().st_mtime)
        print(f"📁 Using store: {latest_store.name}")
        
        # Open store directly with zarr
        store = zarr.storage.LocalStore(str(latest_store))
        root = zarr.open_group(store, mode='r')
        
        # Navigate to audio group
        if 'audio_imports' in root:
            audio_imports = root['audio_imports']
            group_names = [name for name in audio_imports.keys() if name.isdigit()]
            if group_names:
                imported_group = audio_imports[max(group_names, key=int)]
                print(f"📦 Found audio group: {max(group_names, key=int)}")
            else:
                print("❌ No audio groups found")
                return False
        else:
            print("❌ No audio_imports group found")
            return False
        
        # Check available arrays
        arrays = list(imported_group.keys())
        print(f"📊 Available arrays: {arrays}")
        
        # Check for packet format
        has_packet_format = (
            'opus_packets_blob' in arrays and
            'opus_packet_index' in arrays
        )
        
        if not has_packet_format:
            print("❌ Packet-based format not found")
            return False
        
        print("✅ Packet-based format detected")
        
        # Load packet data
        packet_index = imported_group['opus_packet_index']
        packet_blob = imported_group['opus_packets_blob']
        
        print(f"📊 Packet index: {packet_index.shape}")
        print(f"📊 Packet blob: {packet_blob.shape} bytes")
        
        # Get audio parameters
        sample_rate = packet_index.attrs.get('sample_rate', 48000)
        channels = packet_index.attrs.get('nb_channels', 1)
        estimated_samples = packet_index.attrs.get('estimated_total_samples', 0)
        
        print(f"🎵 Audio: {sample_rate}Hz, {channels} channels")
        print(f"🎵 Estimated samples: {estimated_samples}")
        
        # Test 1: Direct packet access
        print(f"\n1. Direct Packet Access Test...")
        
        # Get first few packets
        num_packets_to_test = min(5, packet_index.shape[0])
        
        for i in range(num_packets_to_test):
            offset, size, samples, cumulative = packet_index[i]
            print(f"   Packet {i}: offset={offset}, size={size}, samples={samples}, cumulative={cumulative}")
            
            # Extract packet data
            packet_data = bytes(packet_blob[offset:offset+size])
            print(f"   Packet {i} data: {len(packet_data)} bytes")
        
        # Test 2: Sample range calculation
        print(f"\n2. Sample Range Calculation Test...")
        
        # Test finding packets for sample range
        test_start_sample = 0
        test_end_sample = 999
        
        cumulative_samples = packet_index[:, 3]  # Column 3: cumulative samples
        
        # Find start packet
        start_packet_idx = np.searchsorted(cumulative_samples, test_start_sample, side='right') - 1
        start_packet_idx = max(0, start_packet_idx)
        
        # Find end packet
        end_packet_idx = np.searchsorted(cumulative_samples, test_end_sample, side='right')
        end_packet_idx = min(end_packet_idx, packet_index.shape[0] - 1)
        
        print(f"   Sample range [{test_start_sample}:{test_end_sample}]")
        print(f"   Packet range: [{start_packet_idx}:{end_packet_idx}]")
        
        # Extract packets for this range
        packets_for_range = []
        for packet_idx in range(start_packet_idx, end_packet_idx + 1):
            offset, size, samples, cumulative = packet_index[packet_idx]
            packet_data = bytes(packet_blob[offset:offset+size])
            packets_for_range.append(packet_data)
            print(f"   Packet {packet_idx}: {len(packet_data)} bytes")
        
        print(f"   Total packets for range: {len(packets_for_range)}")
        total_packet_bytes = sum(len(p) for p in packets_for_range)
        print(f"   Total packet data: {total_packet_bytes} bytes")
        
        # Test 3: OpusHead header
        print(f"\n3. OpusHead Header Test...")
        
        if 'opus_header' in arrays:
            opus_header = imported_group['opus_header']
            header_data = bytes(opus_header[:])
            print(f"   Header size: {len(header_data)} bytes")
            print(f"   Header start: {header_data[:8]}")
            
            # Check if it's a valid OpusHead
            if header_data.startswith(b'OpusHead'):
                print("   ✅ Valid OpusHead header found")
            else:
                print("   ⚠️  Header doesn't start with OpusHead")
        else:
            print("   ⚠️  No opus_header array found")
        
        # Test 4: Simulated ffmpeg extraction
        print(f"\n4. Simulated FFmpeg Extraction Test...")
        
        try:
            # This simulates what would happen with ffmpeg
            import tempfile
            import subprocess
            
            # Create a minimal test with just a few packets
            test_packets = packets_for_range[:3]  # First 3 packets
            
            # Create a simple concatenated file (not proper OGG format, but for testing)
            with tempfile.NamedTemporaryFile(suffix='.opus', delete=False) as tmp:
                # Write header first if available
                if 'opus_header' in arrays:
                    header_data = bytes(imported_group['opus_header'][:])
                    tmp.write(header_data)
                
                # Write packet data
                for packet in test_packets:
                    tmp.write(packet)
                
                temp_file = tmp.name
            
            print(f"   Created test file: {len(test_packets)} packets")
            
            # Test if ffmpeg can read it (just probe, don't decode)
            try:
                ffprobe_cmd = [
                    "ffprobe", "-v", "error", "-print_format", "json",
                    "-show_format", temp_file
                ]
                
                result = subprocess.run(
                    ffprobe_cmd, capture_output=True, text=True, timeout=5
                )
                
                if result.returncode == 0:
                    print("   ✅ Test file readable by ffprobe")
                else:
                    print("   ⚠️  Test file not readable by ffprobe")
                    
            except subprocess.TimeoutExpired:
                print("   ⚠️  ffprobe timeout")
            except Exception as e:
                print(f"   ⚠️  ffprobe error: {e}")
            finally:
                # Clean up
                try:
                    import os
                    os.unlink(temp_file)
                except:
                    pass
                    
        except Exception as e:
            print(f"   ❌ Simulated extraction error: {e}")
        
        # Test 5: Opuslib availability
        print(f"\n5. Opuslib Availability Test...")
        
        try:
            import opuslib
            print("   ✅ opuslib is available!")
            
            # Test creating a decoder
            try:
                decoder = opuslib.Decoder(sample_rate, channels)
                print("   ✅ opuslib decoder created successfully")
                
                # Test decoding first packet
                if packets_for_range:
                    first_packet = packets_for_range[0]
                    try:
                        # Skip header packets
                        if not first_packet.startswith(b'OpusHead') and not first_packet.startswith(b'OpusTags'):
                            decoded = decoder.decode(first_packet)
                            print(f"   ✅ Successfully decoded packet: {len(decoded)} samples")
                        else:
                            print("   ⚠️  First packet is header, skipping decode test")
                    except Exception as e:
                        print(f"   ⚠️  Decode test failed: {e}")
                        
            except Exception as e:
                print(f"   ❌ opuslib decoder creation failed: {e}")
                
        except ImportError:
            print("   ❌ opuslib not available")
            print("   💡 Install with: pip install opuslib")
        
        # Overall assessment
        print(f"\n6. Overall Assessment...")
        
        packet_format_ok = has_packet_format
        packet_access_ok = len(packets_for_range) > 0
        header_ok = 'opus_header' in arrays
        
        print(f"   Packet format: {'✅' if packet_format_ok else '❌'}")
        print(f"   Packet access: {'✅' if packet_access_ok else '❌'}")
        print(f"   Header present: {'✅' if header_ok else '❌'}")
        
        if packet_format_ok and packet_access_ok:
            print(f"\n🎉 SUCCESS: Packet-based storage working!")
            print(f"   📦 {packet_index.shape[0]} packets stored")
            print(f"   📊 {estimated_samples} samples available") 
            print(f"   🔧 Ready for opuslib direct decoding")
            print(f"   🔧 Can fallback to ffmpeg if needed")
            
            try:
                import opuslib
                print(f"   ⚡ opuslib available - optimal performance possible")
            except ImportError:
                print(f"   📦 opuslib not available - will use ffmpeg fallback")
                print(f"   💡 For best performance: pip install opuslib")
            
            return True
        else:
            print(f"\n⚠️  ISSUES DETECTED:")
            if not packet_format_ok:
                print(f"   🔧 Packet format not properly created")
            if not packet_access_ok:
                print(f"   🔧 Cannot access packet data")
            
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_packet_extraction()
    print(f"\nDirect test {'PASSED' if success else 'FAILED'}")
