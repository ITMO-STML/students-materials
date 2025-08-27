
# ╒════════════════════════════════════════╕
# | Visualization of metrics              |
# ╘════════════════════════════════════════╛
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("metrics_full.csv")

# Извлекаем dataset и method из названия запуска
df['dataset'] = df['Run'].str.extract(r'^(.*?)_')
df['method'] = df['Run'].str.extract(r'_(baseline|autoprompt)$')

# Удаляем дубликаты (если есть)
df = df.drop_duplicates(subset=['dataset', 'method'], keep='first')

# Pivot-таблицы
pivot_codebleu = df.pivot(index='dataset', columns='method', values='CodeBLEU')
pivot_em = df.pivot(index='dataset', columns='method', values='EM')
pivot_latency = df.pivot(index='dataset', columns='method', values='Latency_ms')

# ── Bar chart: CodeBLEU ─────────────────────
pivot_codebleu.plot(kind='bar', figsize=(8, 5))
plt.ylabel("CodeBLEU")
plt.title("CodeBLEU: Baseline vs AutoPrompt")
plt.xticks(rotation=0)
plt.grid(axis='y'); plt.tight_layout()
plt.savefig("bar_codebleu_comparison.png", dpi=150)
plt.show()

# ── Bar chart: Exact Match ──────────────────
pivot_em.plot(kind='bar', figsize=(8, 5))
plt.ylabel("Exact Match (%)")
plt.title("Exact Match: Baseline vs AutoPrompt")
plt.xticks(rotation=0)
plt.grid(axis='y'); plt.tight_layout()
plt.savefig("bar_em_comparison.png", dpi=150)
plt.show()

# ── Scatter: Latency vs CodeBLEU ────────────
plt.figure(figsize=(7, 5))
for m in df['method'].unique():
    subset = df[df['method'] == m]
    plt.scatter(subset['Latency_ms'], subset['CodeBLEU'], label=m, s=60)

plt.xlabel("Latency (ms)")
plt.ylabel("CodeBLEU")
plt.title("CodeBLEU vs Latency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("scatter_codebleu_vs_latency.png", dpi=150)
plt.show()
