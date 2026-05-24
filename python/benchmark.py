import os
import matplotlib.pyplot as plt

# Размеры матриц
sizes = [100, 200, 400, 800, 1200, 1600, 2000]

# Данные из таблицы (в миллисекундах)
data_ms = {
    1:  [1, 12, 77, 604, 2122, 5340, 10314],
    2:  [0, 5, 39, 306, 1056, 2765, 5274],
    4:  [0, 2, 20, 183, 625, 1455, 2806],
    8:  [0, 1, 12, 181, 456, 1071, 2082],
    12: [0, 1, 10, 131, 433, 1010, 1951],
    16: [0, 1, 8, 66, 207, 480, 921]
}

# Перевод в секунды
data = {k: [v / 1000 for v in values] for k, values in data_ms.items()}

fig, axs = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('MPI Matrix Multiplication Performance (1–16 processes)', 
             fontsize=16, fontweight='bold')

# --- Время (log scale) ---
for processes, times in data.items():
    axs[0, 0].plot(sizes, times, marker='o', linewidth=2.5, label=f'{processes} processes')

axs[0, 0].set_title('Time (log scale)', fontweight='bold')
axs[0, 0].set_ylabel('Time (seconds)')
axs[0, 0].set_yscale('log')
axs[0, 0].grid(True, linestyle='--', alpha=0.5)
axs[0, 0].legend()

# --- Время (linear) ---
for processes, times in data.items():
    axs[0, 1].plot(sizes, times, marker='o', linewidth=2.5, label=f'{processes} processes')

axs[0, 1].set_title('Time (linear scale)', fontweight='bold')
axs[0, 1].set_ylabel('Time (seconds)')
axs[0, 1].grid(True, linestyle='--', alpha=0.5)
axs[0, 1].legend()

# --- Speedup ---
for processes, times in data.items():
    if processes == 1:
        continue
    speedup = [data[1][i] / times[i] if times[i] != 0 else 0 for i in range(len(sizes))]
    axs[1, 0].plot(sizes, speedup, marker='o', linewidth=2.5, label=f'{processes} processes')

axs[1, 0].axhline(y=2, linestyle=':', alpha=0.5, label='Ideal (2)')
axs[1, 0].axhline(y=4, linestyle=':', alpha=0.5, label='Ideal (4)')
axs[1, 0].axhline(y=8, linestyle=':', alpha=0.5, label='Ideal (8)')
axs[1, 0].axhline(y=12, linestyle=':', alpha=0.5, label='Ideal (12)')
axs[1, 0].axhline(y=16, linestyle=':', alpha=0.5, label='Ideal (16)')

axs[1, 0].set_title('Speedup (relative to 1 process)', fontweight='bold')
axs[1, 0].set_ylabel('Speedup')
axs[1, 0].grid(True, linestyle='--', alpha=0.5)
axs[1, 0].legend()

# --- Efficiency ---
for processes, times in data.items():
    if processes == 1:
        continue
    speedup = [data[1][i] / times[i] if times[i] != 0 else 0 for i in range(len(sizes))]
    efficiency = [speedup[i] / processes for i in range(len(sizes))]
    axs[1, 1].plot(sizes, efficiency, marker='o', linewidth=2.5, label=f'{processes} processes')

axs[1, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Ideal efficiency')

axs[1, 1].set_title('Parallel Efficiency', fontweight='bold')
axs[1, 1].set_ylabel('Efficiency')
axs[1, 1].set_ylim(0, 1.2)
axs[1, 1].grid(True, linestyle='--', alpha=0.5)
axs[1, 1].legend()

# Общие настройки
for ax in axs.flat:
    ax.set_xlabel('Matrix size (N × N)')
    ax.set_xticks(sizes)

plt.tight_layout()

# Сохранение
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "..", "src", "benchmark_4plots.png")

os.makedirs(os.path.dirname(save_path), exist_ok=True)

plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Графики сохранены: {os.path.abspath(save_path)}")