#!/usr/bin/env python3
"""
check_system.py - System resource checker for poker_cleaner.py
"""

import multiprocessing
import platform
import sys
import time
import random

def check_cpu_cores():
    """Check available CPU cores"""
    try:
        cores = multiprocessing.cpu_count()
        print(f"🖥️  CPU Cores: {cores}")
        print(f"   Platform: {platform.system()} {platform.machine()}")
        return cores
    except Exception as e:
        print(f"❌ Error checking CPU: {e}")
        return 1

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"🐍 Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠️  Warning: Python 3.8+ recommended")
    else:
        print("✅ Python version OK")

def worker_test(chunk_size):
    """Worker function for performance test"""
    wins = 0
    for _ in range(chunk_size):
        if random.random() > 0.5:
            wins += 1
    return wins

def performance_test(n_cores, iterations=500000):
    """Simple Monte-Carlo performance test"""
    print(f"\n⚡ Performance Test ({iterations:,} iterations):")
    
    # Sequential test
    start = time.time()
    seq_result = worker_test(iterations)
    seq_time = time.time() - start
    
    # Parallel test
    start = time.time()
    chunk_size = iterations // n_cores
    try:
        # Try to set spawn method for compatibility
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set or not supported
    
    with multiprocessing.Pool(processes=n_cores) as pool:
        results = pool.map(worker_test, [chunk_size] * n_cores)
    par_time = time.time() - start
    
    speedup = seq_time / par_time if par_time > 0 else 1
    
    print(f"   Sequential: {seq_time:.2f}s")
    print(f"   Parallel:   {par_time:.2f}s")
    print(f"   Speedup:    {speedup:.1f}x")
    
    if speedup > 0:
        efficiency = (speedup / n_cores) * 100
        print(f"   Efficiency: {efficiency:.1f}%")
    
    return speedup

def main():
    print("=" * 50)
    print("🚀 POKER CLEANER SYSTEM CHECK")
    print("=" * 50)
    
    cores = check_cpu_cores()
    check_python_version()
    
    if cores > 1:
        try:
            speedup = performance_test(cores)
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
    
    # Проверка памяти
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"\n💾 Memory: {memory_gb:.1f} GB total")
    except ImportError:
        print(f"\n💾 Memory: Unable to check (install psutil for details)")
    
    print(f"\n💡 Recommendations:")
    if cores >= 8:
        print("✅ Excellent for poker processing")
    elif cores >= 4:
        print("✅ Good for poker processing")
    else:
        print("⚠️  Consider more CPU cores")
    
    # Оценка времени обработки
    print(f"\n⏱️  Estimated processing times:")
    base_time_per_1000_hands = 60  # секунд на 1000 раздач на одном ядре
    for hands in [1000, 10000, 100000]:
        time_seconds = (hands * base_time_per_1000_hands / 1000) / cores
        if time_seconds < 3600:
            print(f"   {hands:6,} hands: ~{time_seconds/60:.0f} minutes")
        else:
            print(f"   {hands:6,} hands: ~{time_seconds/3600:.1f} hours")
    
    print(f"\n🔧 Usage:")
    print(f"   python poker_cleaner.py input.json output.json")

if __name__ == "__main__":
    main() 