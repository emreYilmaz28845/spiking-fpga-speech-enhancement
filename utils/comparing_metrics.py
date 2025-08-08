import numpy as np
import matplotlib.pyplot as plt

# Metric names
metrics = ['PESQ','STOI','SISDR','IntelSISNR','P808','OVRL','SIG','BAK']

metrics = ['PESQ','STOI','P808','OVRL','SIG','BAK']

# Scores
dfn_scores    = [2.336129, 0.91134708, 13.97896, 13.99431, 3.743795, 3.170853, 3.439706, 4.088344]
fsb_scores    = [2.042506, 0.8931437 , 14.41883, 14.40968, 3.640053, 3.047918, 3.35244 , 3.970883]
spiker_scores = [1.0793  , 0.3032    , -45.5218, -45.4529, 2.1574  , 1.0813  , 1.2188  , 1.2414 ]
new_model     = [1.2    , 0.55      , -38.0   , -38.0   , 2.4     , 1.3     , 1.5     , 2.5    ]
#[1.08    , 0.55      , -38.0   , -38.0   , 2.4     , 1.3     , 1.5     , 2.5    ]
x = np.arange(len(metrics))
width = 0.18                  # 4 × 0.18 ≈ overall bar-cluster width (~0.72)

fig, ax = plt.subplots(figsize=(14, 6))
bars1 = ax.bar(x - 1.5*width, dfn_scores   , width, label='DFN'       , color='#1f77b4')
bars2 = ax.bar(x - 0.5*width, fsb_scores   , width, label='FSB'       , color='#ff7f0e')
bars3 = ax.bar(x + 0.5*width, spiker_scores, width, label='Spiker'    , color='#2ca02c')
bars4 = ax.bar(x + 1.5*width, new_model    , width, label='New Model' , color='#d62728')

# Labels
ax.set_ylabel('Score')
ax.set_title('Metric Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

# Annotate all bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('metric_comparison.png')
plt.show()
