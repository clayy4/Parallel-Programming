import matplotlib.pyplot as plt

sizes = [200, 400, 800, 1200, 1600, 2000]

times_ms = [
    (143.388 + 135.308 + 138.709) / 3,
    (1104.95 + 1115.18 + 1098.59) / 3,
    (9773.66 + 10202 + 10069.5) / 3,
    (35127.6 + 34359 + 34538.8) / 3,
    (81793.6 + 87635.6 + 86340.4) / 3,
    (184532 + 167632 + 179656) / 3
]

times_sec = [t / 1000 for t in times_ms]

plt.plot(sizes, times_sec, marker="o")

plt.xlabel("Matrix size (N × N)")
plt.ylabel("Time (seconds)")
plt.title("Matrix multiplication time complexity")

plt.xticks(sizes)
plt.grid(True)

plt.savefig("benchmark_plot.png")