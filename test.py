import time
import psutil
import torch
import json
import gc
from datetime import datetime
from typing import List, Dict, Tuple

class StreamlinedPerformanceTest:
    def __init__(self):
        self.results = []
        self.start_time = None
        self.cpu_samples = []
        self.memory_samples = []
        
    def run_test(self, generate_func) -> Dict:
        """Run comprehensive performance test"""
        print("OPTIMIZED AI PERFORMANCE TEST")
        print("=" * 50)
        
        # System info
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"System: {cpu_count} cores, {memory_gb:.1f}GB RAM")
        print(f"Device: {device}")
        print(f"PyTorch threads: {torch.get_num_threads()}")
        
        # Test scenarios - optimized for speed validation
        test_cases = [
            ("hi", "Quick response", True),
            ("hello how are you", "Greeting", True),
            ("what is AI", "Simple question", False),
            ("explain machine learning briefly", "Technical query", False),
            ("help me understand python", "Help request", False),
            ("thanks", "Gratitude", True),
        ]
        
        print(f"\n Testing {len(test_cases)} scenarios...")
        
        self.start_time = time.time()
        
        for i, (prompt, desc, expect_fast) in enumerate(test_cases, 1):
            # Monitor system before
            cpu_before = psutil.cpu_percent(interval=0.1)
            memory_before = psutil.virtual_memory().percent
            
            print(f"\nTest {i}: {desc}")
            print(f"Input: '{prompt}'")
            
            # Time the generation
            start = time.time()
            response = generate_func(prompt)
            duration = time.time() - start
            
            # Monitor system after
            cpu_after = psutil.cpu_percent(interval=0.1)
            memory_after = psutil.virtual_memory().percent
            
            # Store samples
            self.cpu_samples.extend([cpu_before, cpu_after])
            self.memory_samples.extend([memory_before, memory_after])
            
            # Evaluate performance
            speed_ok = duration < (0.8 if expect_fast else 2.0)
            cpu_ok = max(cpu_before, cpu_after) < 65
            memory_ok = max(memory_before, memory_after) < 90
            
            result = {
                'test_id': i,
                'prompt': prompt,
                'description': desc,
                'response': response[:60] + "..." if len(response) > 60 else response,
                'duration': duration,
                'expect_fast': expect_fast,
                'cpu_peak': max(cpu_before, cpu_after),
                'memory_peak': max(memory_before, memory_after),
                'speed_ok': speed_ok,
                'cpu_ok': cpu_ok,
                'memory_ok': memory_ok,
                'overall_pass': speed_ok and cpu_ok and memory_ok
            }
            
            self.results.append(result)
            
            # Real-time feedback
            status = " PASS" if result['overall_pass'] else "REVIEW"
            print(f"⏱️ {duration:.3f}s | CPU {result['cpu_peak']:.1f}% | Memory {result['memory_peak']:.1f}% | {status}")
            print(f"Response: {response[:50]}...")
            
            # Brief pause for system stability
            time.sleep(0.3)
        
        return self._generate_report()
    
    def _generate_report(self) -> Dict:
        """Generate comprehensive performance report"""
        total_time = time.time() - self.start_time
        
        # Calculate statistics
        durations = [r['duration'] for r in self.results]
        cpu_peaks = [r['cpu_peak'] for r in self.results]
        memory_peaks = [r['memory_peak'] for r in self.results]
        
        fast_tests = [r for r in self.results if r['expect_fast']]
        complex_tests = [r for r in self.results if not r['expect_fast']]
        
        passed_tests = sum(1 for r in self.results if r['overall_pass'])
        pass_rate = (passed_tests / len(self.results)) * 100
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_duration': total_time,
            'test_count': len(self.results),
            'pass_rate': pass_rate,
            'performance': {
                'avg_response_time': sum(durations) / len(durations),
                'max_response_time': max(durations),
                'fast_avg': sum(r['duration'] for r in fast_tests) / len(fast_tests) if fast_tests else 0,
                'complex_avg': sum(r['duration'] for r in complex_tests) / len(complex_tests) if complex_tests else 0,
                'peak_cpu': max(cpu_peaks),
                'avg_cpu': sum(cpu_peaks) / len(cpu_peaks),
                'peak_memory': max(memory_peaks),
                'avg_memory': sum(memory_peaks) / len(memory_peaks),
            },
            'targets': {
                'cpu_target': 60,
                'memory_target': 85,
                'fast_response_target': 0.8,
                'complex_response_target': 2.0,
            },
            'results': self.results
        }
        
        self._print_report(report)
        return report
    
    def _print_report(self, report: Dict):
        """Print formatted performance report"""
        print("\n" + "=" * 60)
        print("PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        perf = report['performance']
        targets = report['targets']
        
        # Overall summary
        print(f"Overall Performance:")
        print(f"  • Tests passed: {report['pass_rate']:.1f}% ({sum(1 for r in self.results if r['overall_pass'])}/{len(self.results)})")
        print(f"  • Total test time: {report['total_duration']:.2f}s")
        print(f"  • Average response: {perf['avg_response_time']:.3f}s")
        
        # Response time analysis
        print(f"\nResponse Time Analysis:")
        print(f"  • Fast responses: {perf['fast_avg']:.3f}s (target: <{targets['fast_response_target']}s)")
        print(f"  • Complex responses: {perf['complex_avg']:.3f}s (target: <{targets['complex_response_target']}s)")
        print(f"  • Slowest response: {perf['max_response_time']:.3f}s")
        
        # Resource usage
        print(f"\nResource Usage:")
        cpu_status = "✅" if perf['peak_cpu'] < targets['cpu_target'] else "⚠️"
        memory_status = "✅" if perf['peak_memory'] < targets['memory_target'] else "⚠️"
        
        print(f"  • CPU peak: {perf['peak_cpu']:.1f}% (target: <{targets['cpu_target']}%) {cpu_status}")
        print(f"  • CPU average: {perf['avg_cpu']:.1f}%")
        print(f"  • Memory peak: {perf['peak_memory']:.1f}% (target: <{targets['memory_target']}%) {memory_status}")
        print(f"  • Memory average: {perf['avg_memory']:.1f}%")
        
        # Performance rating
        if report['pass_rate'] >= 90 and perf['avg_response_time'] < 1.0:
            rating = "EXCELLENT - All optimization targets met"
        elif report['pass_rate'] >= 75 and perf['avg_response_time'] < 1.5:
            rating = "GOOD - Most targets achieved"
        elif report['pass_rate'] >= 60:
            rating = "ACCEPTABLE - Some optimization needed"
        else:
            rating = "NEEDS WORK - Significant optimization required"
        
        print(f"\nPerformance Rating: {rating}")
        
        # Individual test results
        print(f"\n DETAILED RESULTS:")
        print("-" * 60)
        print(f"{'#':<2} {'Description':<18} {'Time':<8} {'CPU%':<6} {'Mem%':<6} {'Status'}")
        print("-" * 60)
        
        for result in self.results:
            status_icon = "✅" if result['overall_pass'] else "⚠️"
            print(f"{result['test_id']:<2} {result['description'][:17]:<18} "
                  f"{result['duration']:.3f}s{'':<1} {result['cpu_peak']:.1f}{'':<2} "
                  f"{result['memory_peak']:.1f}{'':<2} {status_icon}")
        
        # Recommendations
        self._print_recommendations(report)
    
    def _print_recommendations(self, report: Dict):
        """Print optimization recommendations"""
        print(f"\n RECOMMENDATIONS:")
        print("-" * 40)
        
        perf = report['performance']
        targets = report['targets']
        recommendations = []
        
        # Performance-based recommendations
        if perf['peak_cpu'] > targets['cpu_target']:
            recommendations.append(f" CPU usage high ({perf['peak_cpu']:.1f}%) - reduce PyTorch threads")
        
        if perf['peak_memory'] > targets['memory_target']:
            recommendations.append(f" Memory usage high ({perf['peak_memory']:.1f}%) - increase cache cleanup")
        
        if perf['avg_response_time'] > 1.0:
            recommendations.append(" Slow responses - expand predefined response cache")
        
        if perf['fast_avg'] > targets['fast_response_target']:
            recommendations.append(" Fast responses too slow - optimize quick response detection")
        
        if report['pass_rate'] < 80:
            recommendations.append(" Low pass rate - review generation parameters")
        
        # Positive feedback
        if not recommendations:
            recommendations.append(" Performance looks good!")
            recommendations.append(" Consider running longer stress tests")
        
        for rec in recommendations:
            print(rec)
    
    def save_results(self, report: Dict, filename: str = None):
        """Save test results to JSON file"""
        if filename is None:
            filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n Results saved to: {filename}")
            return filename
        except Exception as e:
            print(f"\n Could not save results: {e}")
            return None

def quick_test(generate_func):
    """Quick 30-second performance validation"""
    print(" QUICK PERFORMANCE CHECK")
    print("-" * 30)
    
    test_prompts = ["hi", "what is AI", "help me"]
    results = []
    
    for prompt in test_prompts:
        start = time.time()
        cpu_before = psutil.cpu_percent(interval=0.1)
        
        response = generate_func(prompt)
        
        duration = time.time() - start
        cpu_after = psutil.cpu_percent(interval=0.1)
        
        results.append({
            'prompt': prompt,
            'duration': duration,
            'cpu_peak': max(cpu_before, cpu_after),
            'response_len': len(response)
        })
        
        status = "✅" if duration < 1.0 else "⚠️"
        print(f"{prompt:<12} | {duration:.3f}s | CPU {max(cpu_before, cpu_after):.1f}% | {status}")
    
    avg_time = sum(r['duration'] for r in results) / len(results)
    max_cpu = max(r['cpu_peak'] for r in results)
    
    print(f"\nAverage: {avg_time:.3f}s | Peak CPU: {max_cpu:.1f}%")
    
    if avg_time < 0.8 and max_cpu < 60:
        print(" Performance: EXCELLENT")
    elif avg_time < 1.2 and max_cpu < 70:
        print(" Performance: GOOD")
    else:
        print(" Performance: NEEDS OPTIMIZATION")
    
    return results

# Example usage with your generate function
def run_performance_test(generate_response_func):
    """Main test runner function"""
    print("Starting optimized performance validation...")
    
    # Quick check first
    print("\n" + "=" * 50)
    quick_results = quick_test(generate_response_func)
    
    # Full test
    print("\n" + "=" * 50)
    tester = StreamlinedPerformanceTest()
    full_report = tester.run_test(generate_response_func)
    
    # Save results
    filename = tester.save_results(full_report)
    
    print(f"\n Testing completed!")
    print(f"Check '{filename}' for detailed results")
    
    return full_report, quick_results

# Usage example:
# If you have your generate_response_optimized function available:
# report, quick = run_performance_test(generate_response_optimized)