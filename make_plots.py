import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("m1_to_m4_results.csv")
df["layout_size"] = df["n_rows"] * df["n_cols"]

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

# 1. M1 vs layout size, both methods, split by wall condition
ax = axes[0, 0]
for wc, ls in [(False, "-"), (True, "--")]:
    sub = df[df["with_walls"] == wc].groupby("layout_size").agg(
        M1_A=("M1_A", "mean"), M1_B=("M1_B", "mean")
    ).reset_index()
    ax.plot(sub["layout_size"], sub["M1_A"], ls, marker="o", color="tab:blue",
            label=f"Method A, walls={wc}")
    ax.plot(sub["layout_size"], sub["M1_B"], ls, marker="s", color="tab:orange",
            label=f"Method B, walls={wc}")
ax.set_xlabel("Layout size (rows x cols)")
ax.set_ylabel("M1 — Entrance Reachability Ratio")
ax.set_title("M1 vs. Layout Size")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# 2. M2 vs wall condition
ax = axes[0, 1]
sub = df.groupby("with_walls")["M2"].agg(["mean", "std"]).reset_index()
ax.bar(sub["with_walls"].astype(str), sub["mean"], yerr=sub["std"], color=["tab:green", "tab:red"])
ax.set_xlabel("with_walls")
ax.set_ylabel("M2 — Reachability Agreement")
ax.set_title("M2 vs. Wall Condition")
ax.grid(alpha=0.3, axis="y")

# 3. M3 distribution (jointly reachable subset), by wall condition
ax = axes[1, 0]
for wc, color in [(False, "tab:green"), (True, "tab:red")]:
    vals = pd.to_numeric(df[(df["with_walls"] == wc)]["M3"], errors="coerce").dropna()
    ax.hist(vals, bins=15, alpha=0.6, label=f"walls={wc}", color=color)
ax.set_xlabel("M3 — Path-Length Consistency")
ax.set_ylabel("Count (configurations)")
ax.set_title("M3 Distribution (jointly reachable zones)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 4. M4 under wall-enabled sweeps, grouped by method
ax = axes[1, 1]
sub = df[df["with_walls"] == True].copy()
m4a = pd.to_numeric(sub["M4_A"], errors="coerce").dropna()
m4b = pd.to_numeric(sub["M4_B"], errors="coerce").dropna()
ax.boxplot([m4a, m4b], labels=["Method A\n(corridor-template)", "Method B\n(centroid-proximity)"])
ax.set_ylabel("M4 — Edge Soundness Validity Rate")
ax.set_title("M4 Under Wall-Enabled Condition")
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("m1_to_m4_results_plots.png", dpi=150)
print("saved m1_to_m4_results_plots.png")
