"""Pick the learning rate per model from the 400-label curves: highest mean (over seeds) of the best validation accuracy."""
import json, sys
from collections import defaultdict
from pathlib import Path
model = sys.argv[1]
best = defaultdict(list)
for f in Path("results/ft_curves").glob(f"{model}-ft400-long-lr*-s*.json"):
    c = json.loads(f.read_text())
    best[c["lr"]].append(max(e["dev_acc"] for e in c["epochs"]))
print(max(best, key=lambda lr: (sum(best[lr]) / len(best[lr]), -lr)))
