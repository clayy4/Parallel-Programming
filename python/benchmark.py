import os
import matplotlib.pyplot as plt


sizes = [200, 400, 800, 1200, 1600, 2000]

data_ms = {
    1: [4.657, 38.4303, 335.977, 1552.19, 4260.47, 16881.6],
    2: [2.825, 21.077, 170.344, 757.334, 2126.66, 7882.45],
    4: [1.55767, 12.162, 101.193, 421.051, 1185.98, 3725.85],
    8: [1.09033, 6.48133, 64.6177, 243.195, 655.875, 2032.9]
}

data = {k: [v/1000 for v in values] for k, values in data_ms.items()}

fig, axs = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('OpenMP Matrix Multiplication Performance (1–8 threads)', fontsize=16)

for threads, times in data.items():
    axs[0, 0].plot(sizes, times, marker='o', linewidth=2.5, label=f'{threads} threads')
axs[0, 0].set_title('Time (log scale)')
axs[0, 0].set_ylabel('Time (seconds)')
axs[0, 0].set_yscale('log')
axs[0, 0].grid(True, linestyle='--', alpha=0.7)
axs[0, 0].legend()

for threads, times in data.items():
    axs[0, 1].plot(sizes, times, marker='o', linewidth=2.5, label=f'{threads} threads')
axs[0, 1].set_title('Time (linear scale)')
axs[0, 1].set_ylabel('Time (seconds)')
axs[0, 1].grid(True, linestyle='--', alpha=0.7)

for threads, times in data.items():
    if threads == 1:
        continue
    speedup = [data[1][i] / times[i] for i in range(len(sizes))]
    axs[1, 0].plot(sizes, speedup, marker='o', linewidth=2.5, label=f'{threads} threads')
axs[1, 0].set_title('Speedup (relative to 1 thread)')
axs[1, 0].set_ylabel('Speedup')
axs[1, 0].grid(True, linestyle='--', alpha=0.7)
axs[1, 0].legend()

for threads, times in data.items():
    if threads == 1:
        continue
    speedup = [data[1][i] / times[i] for i in range(len(sizes))]
    efficiency = [s / threads for s in speedup]
    axs[1, 1].plot(sizes, efficiency, marker='o', linewidth=2.5, label=f'{threads} threads')
axs[1, 1].set_title('Parallel Efficiency (Speedup / Threads)')
axs[1, 1].set_ylabel('Efficiency')
axs[1, 1].grid(True, linestyle='--', alpha=0.7)
axs[1, 1].legend()

for ax in axs.flat:
    ax.set_xlabel('Matrix size (N × N)')
    ax.set_xticks(sizes)

plt.tight_layout()
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "..", "src", "benchmark_4plots.png")

plt.savefig(save_path, dpi=300, bbox_inches='tight')