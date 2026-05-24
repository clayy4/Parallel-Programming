import os
import matplotlib.pyplot as plt

sizes = [200, 400, 800, 1200, 1600, 2000]

# Исходные данные в миллисекундах из вашего бенчмарка CUDA
data_ms = {
    8:  [0.229931, 0.191253, 1.94212, 6.14182, 21.4934, 38.4803],
    16: [0.129547, 0.180384, 1.11190, 5.74917, 13.8371, 19.7083],
    32: [0.121227, 0.255168, 1.36150, 4.31454, 24.0928, 41.8560]
}

# Перевод времени из миллисекунд в секунды
data = {k: [v / 1000 for v in values] for k, values in data_ms.items()}

fig, axs = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('CUDA Shared Memory Matrix Multiplication Performance', fontsize=16)

# 1. Время в логарифмическом масштабе
for tile, times in data.items():
    axs[0, 0].plot(sizes, times, marker='o', linewidth=2.5, label=f'Tile {tile}x{tile}')
axs[0, 0].set_title('Time (log scale)')
axs[0, 0].set_ylabel('Time (seconds)')
axs[0, 0].set_yscale('log')
axs[0, 0].grid(True, linestyle='--', alpha=0.7)
axs[0, 0].legend()

# 2. Время в линейном масштабе
for tile, times in data.items():
    axs[0, 1].plot(sizes, times, marker='o', linewidth=2.5, label=f'Tile {tile}x{tile}')
axs[0, 1].set_title('Time (linear scale)')
axs[0, 1].set_ylabel('Time (seconds)')
axs[0, 1].grid(True, linestyle='--', alpha=0.7)
axs[0, 1].legend()

# 3. Ускорение относительно Tile 8
for tile, times in data.items():
    if tile == 8:
        continue
    speedup = [data[8][i] / times[i] for i in range(len(sizes))]
    axs[1, 0].plot(sizes, speedup, marker='o', linewidth=2.5, label=f'Tile {tile} vs Tile 8')
axs[1, 0].set_title('Speedup (relative to Tile 8)')
axs[1, 0].set_ylabel('Speedup Factor')
axs[1, 0].grid(True, linestyle='--', alpha=0.7)
axs[1, 0].legend()

# 4. Эффективность масштабирования тайла (прирост производительности на единицу изменения размера)
for tile, times in data.items():
    if tile == 8:
        continue
    speedup = [data[8][i] / times[i] for i in range(len(sizes))]
    # Коэффициент изменения размера относительно базового (16/8 = 2, 32/8 = 4)
    scale_factor = tile / 8
    efficiency = [s / scale_factor for s in speedup]
    axs[1, 1].plot(sizes, efficiency, marker='o', linewidth=2.5, label=f'Tile {tile} efficiency')
axs[1, 1].set_title('Tile Scaling Efficiency (Speedup / Scale Factor)')
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