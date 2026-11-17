import csv
from pathlib import Path

from scipy.stats import spearmanr

HERE = Path(__file__).parent
human = {(r["part"], r["folder"]): float(r["rating"]) for r in csv.DictReader(open(HERE / "human_scores.csv"))}
model = [r for r in csv.DictReader(open(HERE / "text_scores.csv")) if (r["part"], r["folder"]) in human]
y = [human[(r["part"], r["folder"])] for r in model]

print(f"{len(model)} images rated by both")
for col in ["text_score", "zs_p_text", "target_sim", "text_level_pred"]:
    print(f"{col:16} Spearman rho = {spearmanr([float(r[col]) for r in model], y)[0]:.2f}")
