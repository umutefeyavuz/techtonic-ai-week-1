import time
import psutil
import torch
import json
from datetime import datetime
from typing import Dict, List

class OptimizedPerformanceTest:
    def __init__(self):
        self.results = []
        
    def run_test(self, generate_func) -> Dict:
        """Streamlined performance test focused on key metrics"""
        print("AI PERFORMANCE TEST")
        print("=" * 50)
        
        # System info
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}, Threads: {torch.get_num_threads()}")
        
        # Minimal test set - only essential cases
        test_cases = [
            ("hi", 0.5, True),  # (prompt, max_time, cached_expected)
            ("what is AI", 1.5, False),
            ("explain python briefly", 2.0, False),
        ]
        
        print(f"\nTesting {len(test_cases)} scenarios...")
        start_time = time.time()
        
        for i, (prompt, max_time, expect_cache) in enumerate(test_cases, 1):
            # Single CPU measurement
            cpu_start = psutil.cpu_percent(interval=0.05)
            
            # Time generation
            gen_start = time.time()
            response = generate_func(prompt)
            duration = time.time() - gen_start
            
            # Single memory check
            memory = psutil.virtual_memory().percent
            
            # Simple pass/fail
            passed = duration < max_time and cpu_start < 70 and memory < 90
            
            self.results.append({
                'id': i,
                'prompt': prompt[:30],
                'response': response[:50],
                'time': duration,
                'cpu': cpu_start,
                'memory': memory,
                'passed': passed
            })
            
            # Immediate feedback
            status = "✓" if passed else "✗"
            print(f"{i}. {duration:.2f}s | CPU {cpu_start:.0f}% | Mem {memory:.0f}% {status}")
            
            # Minimal pause
            time.sleep(0.1)
        
        total_time = time.time() - start_time
        return self._generate_report(total_time)
    
    def _generate_report(self, total_time: float) -> Dict:
        """Generate concise report"""
        times = [r['time'] for r in self.results]
        cpus = [r['cpu'] for r in self.results]
        mems = [r['memory'] for r in self.results]
        passed = sum(1 for r in self.results if r['passed'])
        
        report = {
            'duration': total_time,
            'tests': len(self.results),
            'passed': passed,
            'avg_time': sum(times) / len(times),
            'max_time': max(times),
            'avg_cpu': sum(cpus) / len(cpus),
            'max_cpu': max(cpus),
            'avg_memory': sum(mems) / len(mems),
            'details': self.results
        }
        
        # Print summary
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        print(f"Passed: {passed}/{len(self.results)}")
        print(f"Avg Response: {report['avg_time']:.2f}s")
        print(f"CPU Usage: {report['avg_cpu']:.0f}% avg, {report['max_cpu']:.0f}% peak")
        print(f"Memory: {report['avg_memory']:.0f}%")
        
        # Overall rating
        if passed == len(self.results) and report['avg_time'] < 1.0:
            print("Rating: EXCELLENT")
        elif passed >= len(self.results) - 1:
            print("Rating: GOOD")
        else:
            print("Rating: NEEDS OPTIMIZATION")
        
        return report
    
    def save_results(self, report: Dict) -> str:
        """Save results to JSON"""
        filename = f"test_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nSaved to: {filename}")
            return filename
        except:
            return None

def quick_benchmark(generate_func) -> Dict:
    """Ultra-fast 10-second benchmark"""
    print("QUICK BENCHMARK (10s)")
    print("-" * 30)
    
    prompts = ["hi", "test", "help"]
    times = []
    
    for p in prompts:
        start = time.time()
        _ = generate_func(p)
        t = time.time() - start
        times.append(t)
        print(f"{p:<10} {t:.3f}s")
    
    avg = sum(times) / len(times)
    print(f"\nAverage: {avg:.3f}s")
    print(f"Status: {'GOOD' if avg < 0.8 else 'SLOW'}")
    
    return {'avg': avg, 'times': times}

# Main test runner
def run_performance_test(generate_func):
    """Optimized test runner"""
    # Quick check
    print("\n" + "="*50)
    quick = quick_benchmark(generate_func)
    
    if quick['avg'] > 2.0:
        print("\n⚠ Performance too slow for full test")
        return None, quick
    
    # Full test
    print("\n" + "="*50)
    tester = OptimizedPerformanceTest()
    report = tester.run_test(generate_func)
    tester.save_results(report)
    
    return report, quick