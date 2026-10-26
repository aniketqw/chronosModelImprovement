"""Re-runnable checks behind docs/PATHWAYS_TO_BEAT_CHRONOS.md (needs only numpy).

Run from the repository root:  python docs/evidence/verify_claims.py

It does NOT download or run any Chronos model. It checks four things that can be
checked from the files in this repository and from first principles:

  1. How much the baseline's own aggregate MAE moves between logged runs.
  2. How often the sign of the per-dataset "improvement" flips between runs.
  3. Whether the "adaptive tokenization" input equals the baseline input.
  4. How big the "improvement" of a single 24-step window with 10 samples can be
     when both arms are the SAME model (a toy simulation, Gaussian forecasts).
  5. The floor of the soft-label cross-entropy used in train_improved.py.
"""

import csv
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "DistanceAware"
RUNS = {
    "V1_MODEL/v1": "V1_MODEL/v1",
    "V1_MODEL/v2_soft_label": "V1_MODEL/v2_soft_label",
    "v2_MODEL/V1": "v2_MODEL/V1",
    "v2_MODEL/t3": "v2_MODEL/t3",
}


def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def check_run_noise():
    section("1. Aggregate MAE of the two arms in each logged run")
    orig, da = [], []
    for name, rel in RUNS.items():
        with open(ROOT / rel / "benchmark_results" / "comparison.csv") as f:
            row = next(r for r in csv.DictReader(f) if r["Metric"] == "MAE")
        o, d = float(row["Original"]), float(row["Distance-Aware"])
        orig.append(o)
        da.append(d)
        print(f"{name:26s} original={o:9.4f}  distance-aware={d:9.4f}  gain={(o - d) / o * 100:+.2f}%")
    print(f"\noriginal mean={st.mean(orig):.2f} sd={st.stdev(orig):.2f}  ({st.stdev(orig) / st.mean(orig) * 100:.2f}% of mean)")
    print(f"dist-aware mean={st.mean(da):.2f} sd={st.stdev(da):.2f}")
    sd_rel = st.stdev(orig) / st.mean(orig)
    print(f"two runs of the SAME model differ by 1.96*sqrt(2)*sd = {1.96 * 2 ** 0.5 * sd_rel * 100:.2f}% (95% band)")


def check_sign_flips():
    section("2. Per-dataset MAE improvement (%) in each run (+ means distance-aware wins)")
    table = {}
    for name, rel in RUNS.items():
        with open(ROOT / rel / "benchmark_results" / "per_dataset_results.csv") as f:
            for r in csv.DictReader(f):
                table.setdefault(r["dataset"], {})[name] = float(r["MAE_Improvement_%"])
    print(f"{'dataset':26s}" + "".join(f"{n:>26s}" for n in RUNS))
    flips = 0
    for ds, v in table.items():
        vals = [v[n] for n in RUNS if n in v]
        flip = min(vals) < 0 < max(vals)
        flips += flip
        print(f"{ds:26s}" + "".join(f"{v.get(n, float('nan')):>+26.2f}" for n in RUNS) + ("   <- sign flips" if flip else ""))
    print(f"\n{flips} of {len(table)} datasets change sign between runs")


def check_adaptive_tokenization():
    section("3. 'Adaptive tokenization' input vs baseline input (dm_eval_3/improvement/p1)")

    def baseline(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    def adaptive(x):
        lo, hi = np.percentile(x, [1, 99])
        buf = (hi - lo) * 0.05
        lo, hi = lo - buf, hi + buf
        s = (x - lo) / (hi - lo + 1e-8)
        return (s - s.mean()) / (s.std() + 1e-8)

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(1000):
        x = np.cumsum(rng.normal(size=300)) + rng.uniform(-50, 50)
        worst = max(worst, float(np.abs(baseline(x) - adaptive(x)).max()))
    print(f"max |baseline - adaptive| over 1000 random series = {worst:.2e}")
    print("A z-score does not change under a min-max rescale, so the model sees the same numbers.")


def check_single_window_noise():
    section("4. Same model on both sides: single 24-step window, 10 samples each (toy)")
    rng = np.random.default_rng(0)
    horizon, gains = 24, []
    for _ in range(2000):
        truth = rng.normal(0, 1, horizon)
        a = np.median(rng.normal(0, 1, (10, horizon)), axis=0)
        b = np.median(rng.normal(0, 1, (10, horizon)), axis=0)
        ea, eb = np.abs(a - truth).mean(), np.abs(b - truth).mean()
        gains.append((eb - ea) / eb * 100)
    g = np.array(gains)
    lo, hi = np.percentile(g, [5, 95])
    print(f"relative MAE 'gain' of identical models: sd={g.std():.1f}%  90% interval=[{lo:.1f}%, {hi:.1f}%]")
    print(f"share of trials with |gain| > 12%: {(np.abs(g) > 12).mean():.2f}")


def check_soft_label_floor():
    section("5. Floor of the soft-label cross-entropy (train_improved.py, beta=0.02)")
    beta, n, target = 0.02, 4096, 2048
    d = np.arange(n) - target
    p = np.exp(-beta * d.astype(float) ** 2)
    p /= p.sum()
    h = float(-(p * np.log(p + 1e-30)).sum())
    print(f"label sigma = {(1 / (2 * beta)) ** 0.5:.1f} bins, entropy = {h:.3f} nats")
    print(f"soft term weight 0.5 -> floor {0.5 * h:.3f} of the logged validation loss 1.9125 ({0.5 * h / 1.9125 * 100:.0f}%)")
    print("Cross-entropy against a smoothed label cannot go below the label's entropy.")
    print("\nlabel weight at distance d:   d=1     d=2     d=5     d=10    d=20")
    print("v1 exp(-0.1*d)              " + "  ".join(f"{np.exp(-0.1 * k):.3f}" for k in (1, 2, 5, 10, 20)))
    print("v2 exp(-0.02*d^2)           " + "  ".join(f"{np.exp(-0.02 * k * k):.3f}" for k in (1, 2, 5, 10, 20)))


if __name__ == "__main__":
    check_run_noise()
    check_sign_flips()
    check_adaptive_tokenization()
    check_single_window_noise()
    check_soft_label_floor()
