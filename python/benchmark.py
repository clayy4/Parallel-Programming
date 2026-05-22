import os
import matplotlib.pyplot as plt

# Размеры матриц
sizes = [200, 400, 800, 1200, 1600, 2000]

# Твои реальные данные MPI (в миллисекундах)
data_ms = {
    1: [42.7914, 344.719, 3100.35, 10372.9, 41797.0, 94404.8],
    2: [22.6992, 174.702, 1813.39, 5846.17, 20362.6, 52088.2],
    4: [13.6203, 95.4063, 789.067, 3602.15, 15155.4, 26434.8],
    8: [16.0581, 58.0493, 471.811, 2040.47, 5841.39, 13282.7]
}

# Переводим в секунды для графиков времени
data = {k: [v / 1000 for v in values] for k, values in data_ms.items()}

fig, axs = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('MPI Matrix Multiplication Performance (1–8 processes)', fontsize=16, fontweight='bold')


for processes, times in data.items():
    axs[0, 0].plot(sizes, times, marker='o', linewidth=2.5, label=f'{processes} processes')
axs[0, 0].set_title('Time (log scale)', fontsize=12, fontweight='bold')
axs[0, 0].set_ylabel('Time (seconds)')
axs[0, 0].set_yscale('log')
axs[0, 0].grid(True, linestyle='--', alpha=0.5)
axs[0, 0].legend()

for processes, times in data.items():
    axs[0, 1].plot(sizes, times, marker='o', linewidth=2.5, label=f'{processes} processes')
axs[0, 1].set_title('Time (linear scale)', fontsize=12, fontweight='bold')
axs[0, 1].set_ylabel('Time (seconds)')
axs[0, 1].grid(True, linestyle='--', alpha=0.5)
axs[0, 1].legend()


for processes, times in data.items():
    if processes == 1:
        continue
    speedup = [data[1][i] / times[i] for i in range(len(sizes))]
    axs[1, 0].plot(sizes, speedup, marker='o', linewidth=2.5, label=f'{processes} processes')

# Добавляем линии идеального ускорения для справки
axs[1, 0].axhline(y=2, color='blue', linestyle=':', alpha=0.5, label='Ideal (2 proc)')
axs[1, 0].axhline(y=4, color='green', linestyle=':', alpha=0.5, label='Ideal (4 proc)')
axs[1, 0].axhline(y=8, color='red', linestyle=':', alpha=0.5, label='Ideal (8 proc)')

axs[1, 0].set_title('Speedup (relative to 1 process)', fontsize=12, fontweight='bold')
axs[1, 0].set_ylabel('Speedup Factor')
axs[1, 0].grid(True, linestyle='--', alpha=0.5)
axs[1, 0].legend()


for processes, times in data.items():
    if processes == 1:
        continue
    speedup = [data[1][i] / times[i] for i in range(len(sizes))]
    efficiency = [s / processes for s in speedup]
    axs[1, 1].plot(sizes, efficiency, marker='o', linewidth=2.5, label=f'{processes} processes')

axs[1, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Ideal Efficiency (1.0)')

axs[1, 1].set_title('Parallel Efficiency (Speedup / Processes)', fontsize=12, fontweight='bold')
axs[1, 1].set_ylabel('Efficiency Factor')
axs[1, 1].set_ylim(0, 1.2) 
axs[1, 1].grid(True, linestyle='--', alpha=0.5)
axs[1, 1].legend()

# Общие настройки осей
for ax in axs.flat:
    ax.set_xlabel('Matrix size (N × N)')
    ax.set_xticks(sizes)

plt.tight_layout()

# Сохранение графика
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "..", "src", "benchmark_4plots.png")

# Создаем папку src, если её нет
os.makedirs(os.path.dirname(save_path), exist_ok=True)

plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Потрясающе! Графики успешно сохранены в: {os.path.abspath(save_path)}")