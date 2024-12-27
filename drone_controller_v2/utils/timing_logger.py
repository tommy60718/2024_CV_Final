import time
from collections import deque
import statistics

class TimingLogger:
    def __init__(self, window_size=100):
        self.logs = {}
        self.window_size = window_size
        
    def _ensure_component(self, component):
        if component not in self.logs:
            self.logs[component] = deque(maxlen=self.window_size)
    
    def log_time(self, component, duration):
        """Log the duration for a component"""
        self._ensure_component(component)
        self.logs[component].append(duration)
    
    def get_stats(self, component):
        """Get statistics for a component"""
        if component not in self.logs or not self.logs[component]:
            return None
        
        times = list(self.logs[component])
        return {
            'last': times[-1],
            'avg': statistics.mean(times),
            'min': min(times),
            'max': max(times),
            'median': statistics.median(times)
        }
    
    def print_stats(self):
        """Print statistics for all components"""
        print("\n=== Performance Statistics (ms) ===")
        for component in self.logs:
            stats = self.get_stats(component)
            if stats:
                print(f"\n{component}:")
                print(f"  Last: {stats['last']:.2f}")
                print(f"  Avg: {stats['avg']:.2f}")
                print(f"  Min: {stats['min']:.2f}")
                print(f"  Max: {stats['max']:.2f}")
                print(f"  Median: {stats['median']:.2f}")
        print("================================\n")

timing_logger = TimingLogger() 