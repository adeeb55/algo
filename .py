import numpy as np, time, matplotlib.pyplot as plt, pandas as pd, os, random
from typing import List, Dict

class SortingAnalyzer:
def __init__(self): self.results = {}

def bubble_sort(self, arr): arr = arr.copy(); n = len(arr)
for i in range(n):
swapped = False
for j in range(n - i - 1):
if arr[j] > arr[j+1]:
arr[j], arr[j+1] = arr[j+1], arr[j]; swapped = True
if not swapped: break
return arralg

def merge_sort(self, arr):
if len(arr) <= 1: return arr
mid = len(arr) // 2
return self._merge(self.merge_sort(arr[:mid]), self.merge_sort(arr[mid:]))

def _merge(self, left, right):
result = np.empty(len(left)+len(right), dtype=left.dtype)
i = j = k = 0
while i < len(left) and j < len(right):
result[k] = left[i] if left[i] <= right[j] else right[j]
i += left[i] <= right[j]; j += right[j] < left[i]; k += 1
result[k:] = np.concatenate((left[i:], right[j:]))
return result

def quick_sort(self, arr):
arr = arr.copy(); self._qs(arr, 0, len(arr)-1); return arr

def _qs(self, arr, low, high):
if low < high:
pi = self._partition(arr, low, high)
self._qs(arr, low, pi - 1); self._qs(arr, pi + 1, high)

def _partition(self, arr, low, high):
p = random.randint(low, high); arr[high], arr[p] = arr[p], arr[high]
pivot, i = arr[high], low - 1
for j in range(low, high):
if arr[j] <= pivot:
i += 1; arr[i], arr[j] = arr[j], arr[i]
arr[i+1], arr[high] = arr[high], arr[i+1]
return i + 1

def generate_test_data(self, size, kind='random'):
if kind == 'random': return np.random.randint(1, size*10, size)
elif kind == 'sorted': return np.arange(1, size+1)
elif kind == 'reverse': return np.arange(size, 0, -1)
elif kind == 'partially_sorted':
arr = np.arange(1, size+1); idx = np.random.choice(size, max(1, size//5), False)
np.random.shuffle(arr[idx]); return arr
else: raise ValueError(f"Unknown data type: {kind}")

def measure_execution_time(self, algo, arr, runs=5):
return np.mean([time.perf_counter() - time.perf_counter() + algo(arr.copy()) for _ in range(runs)])

def run_experiments(self, sizes: List[int], types: List[str], runs=5) -> Dict:
algos = {'Bubble Sort': self.bubble_sort, 'Merge Sort': self.merge_sort, 'Quick Sort': self.quick_sort}
res = {'sizes': [], 'data_types': [], 'algorithms': [], 'times': []}
for size in sizes:
for t in types:
data = self.generate_test_data(size, t)
for name, func in algos.items():
res['sizes'].append(size); res['data_types'].append(t)
res['algorithms'].append(name)
res['times'].append(self.measure_execution_time(func, data, runs))
self.results = res; return res

def create_visualizations(self, results=None, save_dir="outputs"):
if results is None: results = self.results
df = pd.DataFrame(results); os.makedirs(save_dir, exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(15, 12)); axes = axes.flatten()
for i, dt in enumerate(df['data_types'].unique()):
ax = axes[i]; sub = df[df['data_types'] == dt]
for alg in sub['algorithms'].unique():
data = sub[sub['algorithms'] == alg]
ax.plot(data['sizes'], data['times'], marker='o', label=alg)
ax.set_title(dt); ax.legend(); ax.set_yscale('log'); ax.grid(True)
plt.tight_layout(); plt.savefig(f"{save_dir}/sorting_performance.png"); plt.close()

def generate_report_table(self, results=None):
if results is None: results = self.results
df = pd.DataFrame(results)
return df.pivot_table(values='times', index=['sizes', 'data_types'], columns='algorithms')

    def verify_correctness(self):
        cases = [np.array([64,34,25,12]), np.array([1]), np.array([]), np.array([3]*4)]
        algos = {'Bubble': self.bubble_sort, 'Merge': self.merge_sort, 'Quick': self.quick_sort}
        for c in cases:
            exp = np.sort(c)
            for name, algo in algos.items():
                assert np.array_equal(algo(c), exp), f"{name} failed"

# Optional: Example usage
# analyzer = SortingAnalyzer()
# analyzer.verify_correctness()
# results = analyzer.run_experiments([1000, 5000], ['random', 'sorted'])
# analyzer.create_visualizations(results)
# print(analyzer.generate_report_table(results)) 
