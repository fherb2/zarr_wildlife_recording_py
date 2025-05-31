#!/usr/bin/env python3
import pathlib
import zarr  # Direct zarr import
from zarrwlr.opus_index_backend import debug_batch_processing

def test_chunk_sizes():
    # Finde letzten Test-Store
    test_results_dir = pathlib.Path(__file__).parent.resolve() / "testresults"
    latest_store = max(test_results_dir.glob("zarr3-store-opus-e2e-*"))
    
    print(f"🧪 Testing chunk sizes with store: {latest_store.name}")
    
    # Öffne Zarr-Store direkt
    store_path = latest_store / "audio_imports"
    print(f"Opening store: {store_path}")
    
    try:
        # Direkte Zarr-Gruppe öffnen
        audio_group = zarr.open_group(str(store_path), mode='r')
        imported_group = audio_group["0"]
        audio_blob_array = imported_group["audio_data_blob_array"]
        
        print(f"✅ Store opened successfully")
        print(f"Audio blob size: {audio_blob_array.shape[0]} bytes")
        print(f"Starting chunk size optimization...")
        
        # Teste Chunk-Größen
        debug_stats = debug_batch_processing(
            imported_group, audio_blob_array, 
            chunk_sizes_mb=[4, 8, 16, 32]
        )
        
        # Zeige Resultate
        print("\n📊 CHUNK SIZE OPTIMIZATION RESULTS:")
        print("=" * 60)
        print("Size | Time    | Pages/sec | Batches | Status")
        print("-" * 60)
        
        for size_mb, stats in debug_stats.items():
            if isinstance(size_mb, int):
                if stats.get('success', False):
                    time_str = f"{stats['processing_time']:.3f}s"
                    pps_str = f"{stats['pages_per_second']:7.1f}"
                    batches_str = f"{stats['batches_created']:2d}"
                    print(f"{size_mb:2d}MB | {time_str:7s} | {pps_str:9s} | {batches_str:7s} | ✅ SUCCESS")
                else:
                    error_str = stats.get('error', 'unknown error')[:30]
                    print(f"{size_mb:2d}MB | ERROR   | -         | -       | ❌ {error_str}")
        
        if 'optimal_chunk_size_mb' in debug_stats:
            optimal = debug_stats['optimal_chunk_size_mb']
            optimal_perf = debug_stats['optimal_performance']
            print("=" * 60)
            print(f"🏆 OPTIMAL CHUNK SIZE: {optimal}MB")
            print(f"   Best performance: {optimal_perf['pages_per_second']:.1f} pages/sec")
            print(f"   Processing time: {optimal_perf['processing_time']:.3f}s")
            print(f"   Batches created: {optimal_perf['batches_created']}")
            
            # Calculate improvement over 8MB baseline
            baseline_8mb = debug_stats.get(8, {})
            if baseline_8mb.get('success', False):
                improvement = optimal_perf['pages_per_second'] / baseline_8mb['pages_per_second']
                print(f"   Improvement over 8MB: {improvement:.2f}x")
        else:
            print("⚠️  No optimal chunk size determined")
        
    except Exception as e:
        print(f"❌ Error opening store: {e}")
        print(f"Store path: {store_path}")
        print(f"Store exists: {store_path.exists()}")
        
        # List available stores for debugging
        print("\nAvailable stores:")
        for store in test_results_dir.glob("zarr3-store-opus-e2e-*"):
            print(f"  {store.name}")

if __name__ == "__main__":
    test_chunk_sizes()