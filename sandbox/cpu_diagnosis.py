"""
CPU-Diagnose für Parallel FLAC Sync-Search
Testet verschiedene Parallelisierungs-Ansätze
"""

import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from typing import List

def cpu_intensive_task(data_chunk: bytes, chunk_id: int) -> dict:
    """CPU-intensive Sync-Suche"""
    start_time = time.time()
    
    sync_count = 0
    operations = 0
    
    # Simuliere Sync-Suche
    for i in range(len(data_chunk) - 1):
        operations += 1
        if i + 1 < len(data_chunk):
            # Simuliere FLAC Sync-Pattern Check
            sync_word = int.from_bytes(data_chunk[i:i+2], 'big')
            if (sync_word & 0xFFFE) == 0xFFF8:
                sync_count += 1
    
    end_time = time.time()
    
    return {
        'chunk_id': chunk_id,
        'sync_count': sync_count,
        'operations': operations,
        'processing_time': end_time - start_time,
        'ops_per_second': operations / (end_time - start_time) if end_time > start_time else 0
    }

def test_sequential(data: bytes, chunk_size: int) -> dict:
    """Sequential Processing Baseline"""
    print("=== Sequential Test ===")
    
    start_time = time.time()
    
    chunks = []
    pos = 0
    while pos < len(data):
        end_pos = min(pos + chunk_size, len(data))
        chunks.append(data[pos:end_pos])
        pos = end_pos
    
    results = []
    for i, chunk in enumerate(chunks):
        result = cpu_intensive_task(chunk, i)
        results.append(result)
    
    total_time = time.time() - start_time
    total_ops = sum(r['operations'] for r in results)
    
    print(f"  Chunks: {len(chunks)}")
    print(f"  Total Time: {total_time:.3f}s")
    print(f"  Total Operations: {total_ops:,}")
    print(f"  Ops/Second: {total_ops/total_time:,.0f}")
    
    return {
        'method': 'sequential',
        'total_time': total_time,
        'total_ops': total_ops,
        'ops_per_second': total_ops/total_time,
        'chunks': len(chunks)
    }

def test_threads(data: bytes, chunk_size: int, max_workers: int) -> dict:
    """Thread-based Parallelization"""
    print(f"=== Threads Test ({max_workers} workers) ===")
    
    start_time = time.time()
    
    chunks = []
    pos = 0
    while pos < len(data):
        end_pos = min(pos + chunk_size, len(data))
        chunks.append(data[pos:end_pos])
        pos = end_pos
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, chunk in enumerate(chunks):
            future = executor.submit(cpu_intensive_task, chunk, i)
            futures.append(future)
        
        for future in futures:
            result = future.result()
            results.append(result)
    
    total_time = time.time() - start_time
    total_ops = sum(r['operations'] for r in results)
    
    print(f"  Chunks: {len(chunks)}")
    print(f"  Workers: {max_workers}")
    print(f"  Total Time: {total_time:.3f}s")
    print(f"  Total Operations: {total_ops:,}")
    print(f"  Ops/Second: {total_ops/total_time:,.0f}")
    
    return {
        'method': f'threads_{max_workers}',
        'total_time': total_time,
        'total_ops': total_ops,
        'ops_per_second': total_ops/total_time,
        'chunks': len(chunks),
        'workers': max_workers
    }

def test_processes(data: bytes, chunk_size: int, max_workers: int) -> dict:
    """Process-based Parallelization (bypasses GIL)"""
    print(f"=== Processes Test ({max_workers} workers) ===")
    
    start_time = time.time()
    
    # Teile Daten in Chunks
    chunks = []
    pos = 0
    while pos < len(data):
        end_pos = min(pos + chunk_size, len(data))
        chunks.append(data[pos:end_pos])
        pos = end_pos
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, chunk in enumerate(chunks):
            future = executor.submit(cpu_intensive_task, chunk, i)
            futures.append(future)
        
        for future in futures:
            result = future.result()
            results.append(result)
    
    total_time = time.time() - start_time
    total_ops = sum(r['operations'] for r in results)
    
    print(f"  Chunks: {len(chunks)}")
    print(f"  Workers: {max_workers}")
    print(f"  Total Time: {total_time:.3f}s")
    print(f"  Total Operations: {total_ops:,}")
    print(f"  Ops/Second: {total_ops/total_time:,.0f}")
    
    return {
        'method': f'processes_{max_workers}',
        'total_time': total_time,
        'total_ops': total_ops,
        'ops_per_second': total_ops/total_time,
        'chunks': len(chunks),
        'workers': max_workers
    }

def create_test_data(size_mb: int) -> bytes:
    """Erstelle Testdaten mit zufälligen Sync-Pattern"""
    print(f"Erstelle {size_mb}MB Testdaten...")
    
    # Erstelle zufällige Daten
    data = np.random.randint(0, 256, size_mb * 1024 * 1024, dtype=np.uint8)
    
    # Füge einige FLAC-ähnliche Sync-Pattern hinzu
    sync_pattern = [0xFF, 0xF8]  # FLAC Sync
    for i in range(0, len(data) - 10000, 10000):  # Alle 10KB ein Sync
        if i + 1 < len(data):
            data[i] = sync_pattern[0]
            data[i + 1] = sync_pattern[1]
    
    print(f"Testdaten erstellt: {len(data)} bytes")
    return data.tobytes()

if __name__ == "__main__":
    print("=== CPU-AUSLASTUNG DIAGNOSE ===\n")
    
    # Teste mit verschiedenen Datengrößen
    test_sizes = [10, 50]  # MB
    chunk_size = 2 * 1024 * 1024  # 2MB Chunks
    
    for size_mb in test_sizes:
        print(f"\n{'='*60}")
        print(f"TEST MIT {size_mb}MB DATEN")
        print(f"{'='*60}")
        
        test_data = create_test_data(size_mb)
        print(f"Chunk-Größe: {chunk_size / 1024 / 1024:.1f}MB")
        
        results = []
        
        # Test 1: Sequential
        sequential_result = test_sequential(test_data, chunk_size)
        results.append(sequential_result)
        
        # Test 2: Threads mit verschiedenen Worker-Anzahlen
        for workers in [1, 2, 4]:
            thread_result = test_threads(test_data, chunk_size, workers)
            results.append(thread_result)
        
        # Test 3: Processes mit verschiedenen Worker-Anzahlen
        for workers in [1, 2, 4]:
            try:
                process_result = test_processes(test_data, chunk_size, workers)
                results.append(process_result)
            except Exception as e:
                print(f"  Process Test mit {workers} Workern fehlgeschlagen: {e}")
        
        # Analyse der Ergebnisse
        print(f"\n{'='*40}")
        print("ERGEBNIS-ANALYSE")
        print(f"{'='*40}")
        
        baseline_ops = sequential_result['ops_per_second']
        
        for result in results:
            speedup = result['ops_per_second'] / baseline_ops
            efficiency = (speedup / result.get('workers', 1)) * 100 if result.get('workers') else 100
            
            print(f"{result['method']:15} | "
                  f"Zeit: {result['total_time']:6.2f}s | "
                  f"Ops/s: {result['ops_per_second']:8,.0f} | "
                  f"Speedup: {speedup:4.2f}x | "
                  f"Effizienz: {efficiency:5.1f}%")
        
        print(f"\n🔍 DIAGNOSE für {size_mb}MB:")
        
        # Finde beste Thread- und Process-Performance
        thread_results = [r for r in results if 'threads' in r['method'] and r.get('workers', 0) > 1]
        process_results = [r for r in results if 'processes' in r['method'] and r.get('workers', 0) > 1]
        
        if thread_results:
            best_thread = max(thread_results, key=lambda x: x['ops_per_second'])
            thread_speedup = best_thread['ops_per_second'] / baseline_ops
            if thread_speedup < 1.2:  # Weniger als 20% Speedup
                print(f"❌ Threads zeigen schlechte Performance (max {thread_speedup:.2f}x) → GIL Problem!")
            else:
                print(f"✅ Threads funktionieren (max {thread_speedup:.2f}x)")
        
        if process_results:
            best_process = max(process_results, key=lambda x: x['ops_per_second'])
            process_speedup = best_process['ops_per_second'] / baseline_ops
            if process_speedup > 1.5:  # Mindestens 50% Speedup
                print(f"✅ Processes zeigen gute Performance ({process_speedup:.2f}x) → Lösung gefunden!")
            else:
                print(f"⚠️ Processes zeigen mäßige Performance ({process_speedup:.2f}x)")
    
    print(f"\n{'='*60}")
    print("EMPFEHLUNG:")
    print("="*60)
    print("Wenn Processes deutlich besser sind als Threads:")
    print("→ Verwende ProcessPoolExecutor statt ThreadPoolExecutor")
    print("→ Das umgeht Python's GIL und nutzt alle CPU-Kerne")
    print("\nWenn beide schlecht sind:")
    print("→ Task ist wahrscheinlich Memory-bound, nicht CPU-bound")
    print("→ Parallelisierung bringt keinen Vorteil")
