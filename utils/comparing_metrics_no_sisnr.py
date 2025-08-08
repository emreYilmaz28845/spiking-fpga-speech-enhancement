import numpy as np
import matplotlib.pyplot as plt

# Original labels and indices you want to keep
all_metrics = ['PESQ','STOI','P808','OVRL','SIG','BAK']

# Scores (same order as all_metrics)
dfn_scores    = [2.336129, 0.91134708, 3.743795, 3.170853, 3.439706, 4.088344]
fsb_scores    = [2.042506, 0.8931437 , 3.640053, 3.047918, 3.35244 , 3.970883]
spiker_scores = [1.0793  , 0.3032    , 2.1574  , 1.0813  , 1.2188  , 1.2414 ]
new_model     = [1.2     , 0.55      , 2.4     , 1.3     , 1.5     , 2.5    ]

# Keep only wanted metrics
metrics       = all_metrics 

# Bar-plot
x, width = np.arange(len(metrics)), 0.18
fig, ax  = plt.subplots(figsize=(14, 6))

ax.bar(x - 1.5*width, dfn_scores   , width, label='DFN'       , color='#1f77b4')
ax.bar(x - 0.5*width, fsb_scores   , width, label='FSB'       , color='#ff7f0e')
ax.bar(x + 0.5*width, spiker_scores, width, label='Spiker'    , color='#2ca02c')
ax.bar(x + 1.5*width, new_model    , width, label='New Model' , color='#d62728')

ax.set_ylabel('Score')
ax.set_title('Metric Comparison (SISDR & IntelSISNR removed)')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

# Annotate
for bars in ax.containers:
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)

plt.tight_layout()
plt.savefig('metric_comparison.png')
plt.show()
